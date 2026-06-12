import { describe, expect, test } from 'bun:test';

import { toolLabel } from './tool-labels.js';

describe('toolLabel', () => {
  test('maps core tools to friendly labels', () => {
    expect(toolLabel('Bash', { command: 'ls' })).toBe('Running a command');
    expect(toolLabel('Read', { file_path: '/workspace/agent/notes.md' })).toBe('Reading notes.md');
    expect(toolLabel('Read', {})).toBe('Reading a file');
    expect(toolLabel('WebSearch', { query: 'x' })).toBe('Searching the web');
    expect(toolLabel('Task', {})).toBe('Running a sub-agent');
  });

  test('maps MCP tools by server', () => {
    expect(toolLabel('mcp__obsidian__search', {})).toBe('Searching the vault');
    expect(toolLabel('mcp__claude_ai_Gmail__search_threads', {})).toBe('Checking email');
    expect(toolLabel('mcp__claude_ai_Google_Calendar__list_events', {})).toBe('Checking the calendar');
    expect(toolLabel('mcp__voss__find_company', {})).toBe('Checking the CRM');
  });

  test('humanizes unknown MCP servers', () => {
    expect(toolLabel('mcp__my_cool_server__do_thing', {})).toBe('Using My cool server');
  });

  test('returns null for user-facing / bookkeeping tools', () => {
    expect(toolLabel('send_message', { text: 'hi' })).toBeNull();
    expect(toolLabel('edit_message', {})).toBeNull();
    expect(toolLabel('ask_question', {})).toBeNull();
    expect(toolLabel('TodoWrite', {})).toBeNull();
  });

  test('returns null for unknown plain tools (generic status stands)', () => {
    expect(toolLabel('SomeFutureTool', {})).toBeNull();
  });
});
