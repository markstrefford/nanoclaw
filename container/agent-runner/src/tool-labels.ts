/**
 * Map a tool invocation to a short, human-readable "what the agent is doing"
 * label, for the live progress-status line (see turn-progress.ts).
 *
 * Returns null for tools that should NOT drive the status line — either
 * because they ARE the user-facing output (send_message, ask_question) or
 * because a generic "Working on it" reads better than the tool name.
 *
 * Labels are intentionally vague about arguments: the status line is a
 * reassurance signal for the user, not an audit log, and tool inputs can
 * contain sensitive content we don't want echoed into the chat.
 */
function basename(p: string): string {
  const parts = String(p).split('/').filter(Boolean);
  return parts[parts.length - 1] || String(p);
}

function prettyServer(raw: string): string {
  const s = raw.replace(/[_-]+/g, ' ').trim();
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : 'a tool';
}

export function toolLabel(name: string, input: unknown): string | null {
  const i = (input ?? {}) as Record<string, unknown>;

  switch (name) {
    case 'Bash':
      return 'Running a command';
    case 'Read':
      return typeof i.file_path === 'string' ? `Reading ${basename(i.file_path)}` : 'Reading a file';
    case 'Edit':
    case 'Write':
    case 'NotebookEdit':
      return 'Editing files';
    case 'Glob':
    case 'Grep':
      return 'Searching files';
    case 'WebSearch':
      return 'Searching the web';
    case 'WebFetch':
      return 'Reading a web page';
    case 'Task':
      return 'Running a sub-agent';
    // User-facing output / bookkeeping — never surface as "working" status.
    case 'send_message':
    case 'send_file':
    case 'edit_message':
    case 'add_reaction':
    case 'ask_question':
    case 'TodoWrite':
      return null;
  }

  // MCP tools are named mcp__<server>__<tool>. Map by server so the user
  // sees "Checking email" rather than "mcp__claude_ai_Gmail__search_threads".
  if (name.startsWith('mcp__')) {
    const server = (name.split('__')[1] ?? '').toLowerCase();
    if (server.includes('obsidian')) return 'Searching the vault';
    if (server.includes('gmail') || server.includes('email')) return 'Checking email';
    if (server.includes('calendar') || server.includes('caldav')) return 'Checking the calendar';
    if (server.includes('voss') || server.includes('crm')) return 'Checking the CRM';
    if (server.includes('signal') || server.includes('strata')) return 'Checking SignalStrata';
    if (server.includes('slack')) return 'Checking Slack';
    if (server.includes('notion')) return 'Checking Notion';
    if (server.includes('drive')) return 'Checking Google Drive';
    return `Using ${prettyServer(name.split('__')[1] ?? '')}`;
  }

  // Unknown tool — let the generic "Working on it" stand.
  return null;
}
