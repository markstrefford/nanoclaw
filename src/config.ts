import os from 'os';
import path from 'path';

import { readEnvFile } from './env.js';
import { getContainerImageBase, getDefaultContainerImage, getInstallSlug } from './install-slug.js';
import { isValidTimezone } from './timezone.js';

// Read config values from .env (falls back to process.env).
const envConfig = readEnvFile([
  'ASSISTANT_NAME',
  'ASSISTANT_HAS_OWN_NUMBER',
  'ONECLI_URL',
  'ONECLI_API_KEY',
  'WHISPER_BIN',
  'WHISPER_MODEL',
  'TZ',
  'COST_REPORT_TARGET',
  'COST_REPORT_HOUR',
  'MEDIA_ARCHIVE_DIR',
  'MEDIA_ARCHIVE_MOUNT',
]);

export const COST_REPORT_TARGET = process.env.COST_REPORT_TARGET || envConfig.COST_REPORT_TARGET || '';
export const COST_REPORT_HOUR = process.env.COST_REPORT_HOUR || envConfig.COST_REPORT_HOUR || '';

export const ASSISTANT_NAME = process.env.ASSISTANT_NAME || envConfig.ASSISTANT_NAME || 'Andy';
export const ASSISTANT_HAS_OWN_NUMBER =
  (process.env.ASSISTANT_HAS_OWN_NUMBER || envConfig.ASSISTANT_HAS_OWN_NUMBER) === 'true';

// Absolute paths needed for container mounts
const PROJECT_ROOT = process.cwd();
const HOME_DIR = process.env.HOME || os.homedir();

// Mount security: allowlist stored OUTSIDE project root, never mounted into containers
export const MOUNT_ALLOWLIST_PATH = path.join(HOME_DIR, '.config', 'nanoclaw', 'mount-allowlist.json');
export const SENDER_ALLOWLIST_PATH = path.join(HOME_DIR, '.config', 'nanoclaw', 'sender-allowlist.json');
export const STORE_DIR = path.resolve(PROJECT_ROOT, 'store');
export const GROUPS_DIR = path.resolve(PROJECT_ROOT, 'groups');
export const DATA_DIR = path.resolve(PROJECT_ROOT, 'data');

// Per-checkout image tag so two installs on the same host don't share
// `nanoclaw-agent:latest` and clobber each other on rebuild.
export const CONTAINER_IMAGE_BASE = process.env.CONTAINER_IMAGE_BASE || getContainerImageBase(PROJECT_ROOT);
export const CONTAINER_IMAGE = process.env.CONTAINER_IMAGE || getDefaultContainerImage(PROJECT_ROOT);
// Install slug — stamped onto every spawned container via --label so
// cleanupOrphans only reaps containers from this install, not peers.
export const INSTALL_SLUG = getInstallSlug(PROJECT_ROOT);
export const CONTAINER_INSTALL_LABEL = `nanoclaw-install=${INSTALL_SLUG}`;
export const CONTAINER_TIMEOUT = parseInt(process.env.CONTAINER_TIMEOUT || '1800000', 10);
export const CONTAINER_MAX_OUTPUT_SIZE = parseInt(process.env.CONTAINER_MAX_OUTPUT_SIZE || '10485760', 10); // 10MB default
export const ONECLI_URL = process.env.ONECLI_URL || envConfig.ONECLI_URL;
export const ONECLI_API_KEY = process.env.ONECLI_API_KEY || envConfig.ONECLI_API_KEY;
export const MAX_MESSAGES_PER_PROMPT = Math.max(1, parseInt(process.env.MAX_MESSAGES_PER_PROMPT || '10', 10) || 10);
export const IDLE_TIMEOUT = parseInt(process.env.IDLE_TIMEOUT || '1800000', 10); // 30min default — how long to keep container alive after last result
export const MAX_CONCURRENT_CONTAINERS = Math.max(1, parseInt(process.env.MAX_CONCURRENT_CONTAINERS || '5', 10) || 5);

// Local voice transcription (whisper.cpp on the host). Ported from v1.
// WHISPER_BIN: the whisper.cpp CLI (Homebrew installs it as `whisper-cli`).
// WHISPER_MODEL: path to a ggml model file (default: data/models/ggml-base.bin).
export const WHISPER_BIN = process.env.WHISPER_BIN || envConfig.WHISPER_BIN || 'whisper-cli';
export const WHISPER_MODEL =
  process.env.WHISPER_MODEL || envConfig.WHISPER_MODEL || path.resolve(PROJECT_ROOT, 'data', 'models', 'ggml-base.bin');

// Inbound media archive (images + voice notes) → a persistent folder, by
// design the Obsidian vault's `raw/` so both agents see drops via the existing
// vault mount. Feature is OFF unless MEDIA_ARCHIVE_DIR is set.
//   MEDIA_ARCHIVE_DIR   — host path to the `raw` dir (e.g. ".../10x Mark/raw").
//                         images/ and voicenotes/ subfolders are created under it.
//   MEDIA_ARCHIVE_MOUNT — the same dir's path *inside the container*, relative
//                         to /workspace (default matches the vault mount), used
//                         to tell the agent where to Read the file.
export const MEDIA_ARCHIVE_DIR = process.env.MEDIA_ARCHIVE_DIR || envConfig.MEDIA_ARCHIVE_DIR || '';
export const MEDIA_ARCHIVE_MOUNT =
  process.env.MEDIA_ARCHIVE_MOUNT || envConfig.MEDIA_ARCHIVE_MOUNT || 'extra/obsidian-vault/raw';

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function buildTriggerPattern(trigger: string): RegExp {
  return new RegExp(`^${escapeRegex(trigger.trim())}\\b`, 'i');
}

export const DEFAULT_TRIGGER = `@${ASSISTANT_NAME}`;

export function getTriggerPattern(trigger?: string): RegExp {
  const normalizedTrigger = trigger?.trim();
  return buildTriggerPattern(normalizedTrigger || DEFAULT_TRIGGER);
}

export const TRIGGER_PATTERN = buildTriggerPattern(DEFAULT_TRIGGER);

// Timezone for scheduled tasks, message formatting, etc.
// Validates each candidate is a real IANA identifier before accepting.
function resolveConfigTimezone(): string {
  const candidates = [process.env.TZ, envConfig.TZ, Intl.DateTimeFormat().resolvedOptions().timeZone];
  for (const tz of candidates) {
    if (tz && isValidTimezone(tz)) return tz;
  }
  return 'UTC';
}
export const TIMEZONE = resolveConfigTimezone();
