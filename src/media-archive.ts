/**
 * Inbound media archive.
 *
 * Persists incoming images and voice notes into a long-lived folder — by design
 * the Obsidian vault's `raw/` directory, which is already bind-mounted into the
 * agent containers, so a file written here on the host is directly readable by
 * the agent at `/workspace/<MEDIA_ARCHIVE_MOUNT>/...`. The agent "sees" images
 * via its Read tool (the vault mount is the vision surface); voice notes are
 * transcribed elsewhere and archived here as audio + an Obsidian-native note.
 *
 * Layout (under MEDIA_ARCHIVE_DIR):
 *   images/<ts>-<sender>-<msgid>.<ext>
 *   voicenotes/<ts>-<sender>-<msgid>.<ext>   (+ a sibling .md note)
 *
 * Feature is a no-op unless MEDIA_ARCHIVE_DIR is configured, so installs without
 * a vault keep the existing session-inbox behaviour. `sender` and `messageId`
 * are untrusted input; both are sanitised to `[a-z0-9-]` before they reach a
 * filename, and writes use the `wx` flag (never follow a symlink or clobber).
 */
import fs from 'fs';
import path from 'path';

import { MEDIA_ARCHIVE_DIR, MEDIA_ARCHIVE_MOUNT } from './config.js';
import { log } from './log.js';

export interface ArchiveMeta {
  sender?: string | null;
  messageId: string;
  mimeType?: string | null;
  /** Source platform, recorded in the voice-note note frontmatter. */
  source?: string | null;
  timestamp?: Date;
}

export interface ArchivedFile {
  /** Absolute host path written. */
  hostPath: string;
  /** Path relative to /workspace — what the agent Reads inside the container. */
  containerPath: string;
  /** Bare filename actually used (may differ from request on collision). */
  filename: string;
}

export function isMediaArchiveEnabled(): boolean {
  return Boolean(MEDIA_ARCHIVE_DIR);
}

export function isImageAttachment(type?: string | null, mimeType?: string | null): boolean {
  const t = (type || '').toLowerCase();
  if (t === 'image' || t === 'photo') return true;
  return Boolean(mimeType && mimeType.toLowerCase().startsWith('image/'));
}

function extFromMime(mime: string | null | undefined, fallback: string): string {
  const m = (mime || '').toLowerCase();
  if (m.includes('jpeg') || m.includes('jpg')) return 'jpg';
  if (m.includes('png')) return 'png';
  if (m.includes('gif')) return 'gif';
  if (m.includes('webp')) return 'webp';
  if (m.includes('heic')) return 'heic';
  if (m.includes('ogg')) return 'ogg';
  if (m.includes('mpeg') || m.includes('mp3')) return 'mp3';
  if (m.includes('m4a') || m.includes('mp4')) return 'm4a';
  if (m.includes('wav')) return 'wav';
  const sub = m.split('/')[1]?.replace(/[^a-z0-9]/g, '');
  return sub || fallback;
}

function sanitize(s: string, max = 40): string {
  return (
    s
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, max) || 'x'
  );
}

/** Colon-free, sortable, human-readable stamp (Obsidian/macOS filename-safe). */
function stamp(d: Date): string {
  const p = (n: number): string => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}-${p(d.getSeconds())}`;
}

function buildName(meta: ArchiveMeta, ext: string): string {
  const ts = stamp(meta.timestamp ?? new Date());
  const sender = sanitize(meta.sender || 'unknown');
  const mid = sanitize(meta.messageId, 8);
  return `${ts}-${sender}-${mid}.${ext}`;
}

/**
 * Create `dir` and write `data` to `filename` within it, never following a
 * symlink or overwriting an existing file (wx). On collision, appends `-N`.
 * Returns the absolute path actually written.
 */
function writeUnique(dir: string, filename: string, data: Buffer | string): string {
  fs.mkdirSync(dir, { recursive: true });
  const ext = path.extname(filename);
  const base = filename.slice(0, filename.length - ext.length);
  for (let attempt = 0; attempt <= 50; attempt++) {
    const name = attempt === 0 ? filename : `${base}-${attempt}${ext}`;
    const target = path.join(dir, name);
    try {
      fs.writeFileSync(target, data, { flag: 'wx' });
      return target;
    } catch (err) {
      const e = err as NodeJS.ErrnoException;
      if (e.code === 'EEXIST') continue;
      throw err;
    }
  }
  throw new Error(`media-archive: could not find a free filename for ${filename}`);
}

function toContainerPath(subdir: string, filename: string): string {
  return path.posix.join(MEDIA_ARCHIVE_MOUNT, subdir, filename);
}

/** Write an image into raw/images/. Returns null if disabled or on failure. */
export function archiveImage(buffer: Buffer, meta: ArchiveMeta): ArchivedFile | null {
  if (!MEDIA_ARCHIVE_DIR) return null;
  try {
    const filename = buildName(meta, extFromMime(meta.mimeType, 'img'));
    const hostPath = writeUnique(path.join(MEDIA_ARCHIVE_DIR, 'images'), filename, buffer);
    const actual = path.basename(hostPath);
    log.debug('Archived image to vault', { hostPath, size: buffer.length });
    return { hostPath, filename: actual, containerPath: toContainerPath('images', actual) };
  } catch (err) {
    log.warn('Failed to archive image to vault', { err });
    return null;
  }
}

/**
 * Write a voice note's audio into raw/voicenotes/ alongside an Obsidian note
 * that embeds the audio (`![[...]]`) and carries the transcript as its body.
 * Returns the audio ArchivedFile, or null if disabled / on failure.
 */
export function archiveVoiceNote(
  buffer: Buffer,
  transcript: string | null,
  meta: ArchiveMeta,
): ArchivedFile | null {
  if (!MEDIA_ARCHIVE_DIR) return null;
  try {
    const dir = path.join(MEDIA_ARCHIVE_DIR, 'voicenotes');
    const audioHostPath = writeUnique(dir, buildName(meta, extFromMime(meta.mimeType, 'ogg')), buffer);
    const audioName = path.basename(audioHostPath);

    const received = (meta.timestamp ?? new Date()).toISOString();
    const from = (meta.sender || 'unknown').replace(/[\r\n]+/g, ' ');
    const note = [
      '---',
      'type: voicenote',
      `from: ${from}`,
      `received: ${received}`,
      `audio: ${audioName}`,
      `source: ${meta.source || 'unknown'}`,
      '---',
      '',
      `![[${audioName}]]`,
      '',
      transcript && transcript.trim() ? transcript.trim() : '_(transcription unavailable)_',
      '',
    ].join('\n');
    const noteName = audioName.slice(0, audioName.length - path.extname(audioName).length) + '.md';
    writeUnique(dir, noteName, note);

    log.debug('Archived voice note to vault', { audioHostPath, transcribed: Boolean(transcript) });
    return { hostPath: audioHostPath, filename: audioName, containerPath: toContainerPath('voicenotes', audioName) };
  } catch (err) {
    log.warn('Failed to archive voice note to vault', { err });
    return null;
  }
}
