import { execFile } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { WHISPER_BIN, WHISPER_MODEL } from './config.js';
import { log } from './log.js';

const execFileAsync = promisify(execFile);

/**
 * Transcribe an audio file using local whisper.cpp.
 * Returns the transcript text, or null if transcription is unavailable/fails.
 * The input file can be any format ffmpeg supports (OGG, MP3, M4A, etc.).
 *
 * Host-side feature (ported from v1): whisper-cli + ffmpeg must be installed on
 * the host. Fails safe — any missing binary/model returns null and the caller
 * just keeps the raw audio attachment.
 */
export async function transcribeAudio(inputPath: string): Promise<string | null> {
  const modelPath = path.resolve(WHISPER_MODEL);
  if (!fs.existsSync(modelPath)) {
    log.warn('Whisper model not found, skipping transcription', { modelPath });
    return null;
  }

  const wavPath = path.join(os.tmpdir(), `nanoclaw-voice-${Date.now()}-${Math.random().toString(36).slice(2)}.wav`);

  try {
    // Convert to 16kHz mono WAV (whisper.cpp requirement).
    // Use full path — launchd PATH doesn't include /opt/homebrew/bin.
    const ffmpegBin = fs.existsSync('/opt/homebrew/bin/ffmpeg') ? '/opt/homebrew/bin/ffmpeg' : 'ffmpeg';
    await execFileAsync(ffmpegBin, ['-i', inputPath, '-ar', '16000', '-ac', '1', '-f', 'wav', '-y', wavPath], {
      timeout: 30_000,
    });

    // Run whisper.cpp (no timestamps, plain text output).
    const { stdout } = await execFileAsync(WHISPER_BIN, ['-m', modelPath, '-f', wavPath, '--no-timestamps', '-nt'], {
      timeout: 60_000,
    });

    const transcript = stdout.trim();
    if (!transcript) {
      log.warn('Whisper produced empty transcript');
      return null;
    }

    log.info('Transcribed voice message', { chars: transcript.length });
    return transcript;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } catch (err: any) {
    if (err.code === 'ENOENT') {
      log.warn('Transcription binary not found, skipping', { bin: err.path });
    } else {
      log.warn('whisper.cpp transcription failed', { err: err.message });
    }
    return null;
  } finally {
    try {
      fs.unlinkSync(wavPath);
    } catch {
      /* ignore */
    }
  }
}
