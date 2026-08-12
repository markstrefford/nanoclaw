import { describe, it, expect } from 'bun:test';
import { healthGateMcpServers, type McpServerSpec } from './mcp-health.js';

// Mock MCP servers as one-liner scripts run by the current runtime. The probe
// drives initialize -> tools/list; each mock plays a role in that handshake.
const RT = process.execPath;

const HEALTHY = `
let buf='';process.stdin.on('data',d=>{buf+=d;let i;while((i=buf.indexOf('\\n'))>=0){const l=buf.slice(0,i).trim();buf=buf.slice(i+1);if(!l)continue;const m=JSON.parse(l);if(m.method==='initialize')process.stdout.write(JSON.stringify({jsonrpc:'2.0',id:1,result:{capabilities:{}}})+'\\n');if(m.method==='tools/list')process.stdout.write(JSON.stringify({jsonrpc:'2.0',id:2,result:{tools:[]}})+'\\n');}});
setInterval(()=>{},1000);
`;
// Answers initialize but never answers tools/list — must be dropped.
const HALF = `
let buf='';process.stdin.on('data',d=>{buf+=d;let i;while((i=buf.indexOf('\\n'))>=0){const l=buf.slice(0,i).trim();buf=buf.slice(i+1);if(!l)continue;const m=JSON.parse(l);if(m.method==='initialize')process.stdout.write(JSON.stringify({jsonrpc:'2.0',id:1,result:{}})+'\\n');}});
setInterval(()=>{},1000);
`;
// Healthy, but only answers after a delay — caldav-mcp's shape (it logs into
// iCloud before connecting its transport), which the SDK sees as still pending.
const SLOW = `
setTimeout(()=>{${HEALTHY}},1200);
setInterval(()=>{},1000);
`;
const HANG = `setInterval(()=>{},1000);`; // never responds
const CRASH = `process.exit(1);`;

const spec = (script: string): McpServerSpec => ({ command: RT, args: ['-e', script], env: {} });

describe('healthGateMcpServers', () => {
  it('keeps a server that completes initialize + tools/list', async () => {
    const r = await healthGateMcpServers({ good: spec(HEALTHY) }, { timeoutMs: 2000 });
    expect(Object.keys(r.healthy)).toEqual(['good']);
    expect(r.dropped).toEqual([]);
  });

  it('drops a server that hangs on startup', async () => {
    const r = await healthGateMcpServers({ stuck: spec(HANG) }, { timeoutMs: 500 });
    expect(r.healthy.stuck).toBeUndefined();
    expect(r.dropped.map((d) => d.name)).toEqual(['stuck']);
  });

  it('drops a server that crashes before responding', async () => {
    const r = await healthGateMcpServers({ dead: spec(CRASH) }, { timeoutMs: 2000 });
    expect(r.healthy.dead).toBeUndefined();
    expect(r.dropped.map((d) => d.name)).toEqual(['dead']);
  });

  it('drops a server that initializes but hangs on tools/list', async () => {
    const r = await healthGateMcpServers({ half: spec(HALF) }, { timeoutMs: 500 });
    expect(r.healthy.half).toBeUndefined();
    expect(r.dropped[0].name).toBe('half');
  });

  it('isolates failures: one bad server does not drop the healthy ones', async () => {
    const r = await healthGateMcpServers(
      { good: spec(HEALTHY), stuck: spec(HANG) },
      { timeoutMs: 800 },
    );
    expect(Object.keys(r.healthy)).toEqual(['good']);
    expect(r.dropped.map((d) => d.name)).toEqual(['stuck']);
  });

  it('flags a healthy but slow server without dropping it', async () => {
    const r = await healthGateMcpServers({ lazy: spec(SLOW) }, { timeoutMs: 5000 });
    expect(Object.keys(r.healthy)).toEqual(['lazy']);
    expect(r.dropped).toEqual([]);
    expect(r.slow.map((s) => s.name)).toEqual(['lazy']);
  });

  it('does not flag a server that answers promptly', async () => {
    const r = await healthGateMcpServers({ good: spec(HEALTHY) }, { timeoutMs: 2000 });
    expect(r.slow).toEqual([]);
  });

  it('passes trusted (skip) servers through without probing', async () => {
    // CRASH would be dropped if probed; skipping keeps it.
    const r = await healthGateMcpServers({ nanoclaw: spec(CRASH) }, { skip: ['nanoclaw'], timeoutMs: 500 });
    expect(Object.keys(r.healthy)).toEqual(['nanoclaw']);
    expect(r.dropped).toEqual([]);
  });
});
