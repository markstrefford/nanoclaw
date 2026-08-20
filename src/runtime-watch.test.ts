/**
 * Tests for the container-runtime watch — the degraded-mode path that replaced
 * "exit on a failed `docker info`". The behaviour that matters: the host stays
 * up, the owner is told once per outage (and again on a long one), and
 * recovery is detected without a restart.
 */
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';

vi.mock('./log.js', () => ({
  log: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn(), fatal: vi.fn() },
}));

const mockIsUp = vi.fn();
const mockCleanupOrphans = vi.fn();
const mockStart = vi.fn(() => true);
vi.mock('./container-runtime.js', () => ({
  isContainerRuntimeUp: () => mockIsUp(),
  cleanupOrphans: () => mockCleanupOrphans(),
  startContainerRuntime: () => mockStart(),
}));

const mockNotify = vi.fn(async (_text: string) => true);
vi.mock('./modules/approvals/primitive.js', () => ({
  notifyOwnerGlobal: (text: string) => mockNotify(text),
}));

import {
  _resetRuntimeWatchForTesting,
  checkContainerRuntime,
  decideRuntimeTransition,
  formatDuration,
  isRuntimeDown,
} from './runtime-watch.js';

const HOUR = 60 * 60 * 1000;
const BASE = Date.parse('2026-08-17T08:45:00.000Z');

beforeEach(() => {
  vi.clearAllMocks();
  mockStart.mockReturnValue(true);
  _resetRuntimeWatchForTesting();
  vi.useFakeTimers();
  vi.setSystemTime(BASE);
});

afterEach(() => {
  vi.useRealTimers();
});

describe('decideRuntimeTransition', () => {
  it('treats a first probe that finds the runtime down as went-down', () => {
    expect(decideRuntimeTransition(null, false, BASE, 0)).toBe('went-down');
  });

  it('stays quiet on a first probe that finds the runtime up', () => {
    expect(decideRuntimeTransition(null, true, BASE, 0)).toBe('unchanged');
  });

  it('reports recovery only from a known-down state', () => {
    expect(decideRuntimeTransition(false, true, BASE, 0)).toBe('came-up');
    expect(decideRuntimeTransition(true, true, BASE, 0)).toBe('unchanged');
  });

  it('does not re-alert within the re-alert window', () => {
    expect(decideRuntimeTransition(false, false, BASE + HOUR, BASE)).toBe('unchanged');
  });

  it('re-alerts once the window has elapsed', () => {
    expect(decideRuntimeTransition(false, false, BASE + 6 * HOUR, BASE)).toBe('still-down');
  });
});

describe('formatDuration', () => {
  it('scales from minutes to days', () => {
    expect(formatDuration(5 * 60_000)).toBe('5 min');
    expect(formatDuration(90 * 60_000)).toBe('1.5 h');
    expect(formatDuration(28.5 * HOUR)).toBe('1.2 days');
  });
});

describe('checkContainerRuntime', () => {
  it('alerts once when the runtime goes down and reports it as down', async () => {
    mockIsUp.mockReturnValue(false);

    expect(await checkContainerRuntime()).toBe(false);
    expect(isRuntimeDown()).toBe(true);
    expect(mockNotify).toHaveBeenCalledTimes(1);
    expect(mockNotify.mock.calls[0][0]).toContain('Docker');

    // Next tick, still down, inside the re-alert window — no second message.
    vi.setSystemTime(BASE + 60_000);
    await checkContainerRuntime();
    expect(mockNotify).toHaveBeenCalledTimes(1);
  });

  it('re-alerts after the window while the outage continues', async () => {
    mockIsUp.mockReturnValue(false);
    await checkContainerRuntime();

    vi.setSystemTime(BASE + 6 * HOUR);
    await checkContainerRuntime();

    expect(mockNotify).toHaveBeenCalledTimes(2);
    expect(mockNotify.mock.calls[1][0]).toContain('6.0 h');
  });

  it('cleans up orphans and reports recovery without a restart', async () => {
    mockIsUp.mockReturnValue(false);
    await checkContainerRuntime();
    expect(mockCleanupOrphans).not.toHaveBeenCalled();

    mockIsUp.mockReturnValue(true);
    vi.setSystemTime(BASE + 28.5 * HOUR);

    expect(await checkContainerRuntime()).toBe(true);
    expect(isRuntimeDown()).toBe(false);
    expect(mockCleanupOrphans).toHaveBeenCalledTimes(1);
    expect(mockNotify).toHaveBeenCalledTimes(2);
    expect(mockNotify.mock.calls[1][0]).toContain('1.2 days');
  });

  it('says nothing at all when the runtime is up throughout', async () => {
    mockIsUp.mockReturnValue(true);

    await checkContainerRuntime();
    vi.setSystemTime(BASE + 60_000);
    await checkContainerRuntime();

    expect(mockNotify).not.toHaveBeenCalled();
    expect(mockCleanupOrphans).not.toHaveBeenCalled();
    expect(mockStart).not.toHaveBeenCalled();
  });
});

describe('auto-start', () => {
  it('tries to start the runtime on the way down and says so', async () => {
    mockIsUp.mockReturnValue(false);

    await checkContainerRuntime();

    expect(mockStart).toHaveBeenCalledTimes(1);
    expect(mockNotify.mock.calls[0][0]).toContain("I've asked Docker to start");
  });

  it('does not relaunch on every 60s tick while Docker is booting', async () => {
    mockIsUp.mockReturnValue(false);
    await checkContainerRuntime();

    for (let i = 1; i <= 5; i++) {
      vi.setSystemTime(BASE + i * 60_000);
      await checkContainerRuntime();
    }

    expect(mockStart).toHaveBeenCalledTimes(1);
  });

  it('retries the launch once the retry interval has passed', async () => {
    mockIsUp.mockReturnValue(false);
    await checkContainerRuntime();

    vi.setSystemTime(BASE + 10 * 60_000);
    await checkContainerRuntime();

    expect(mockStart).toHaveBeenCalledTimes(2);
  });

  it('tells the owner it needs a hand when the launch is not possible', async () => {
    mockIsUp.mockReturnValue(false);
    mockStart.mockReturnValue(false);

    await checkContainerRuntime();

    expect(mockNotify.mock.calls[0][0]).toContain('needs a hand');
  });
});
