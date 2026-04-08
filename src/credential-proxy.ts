/**
 * Credential proxy for container isolation.
 * Containers connect here instead of directly to the Anthropic API.
 * The proxy injects real credentials so containers never see them.
 *
 * Two auth modes:
 *   API key:  Proxy injects x-api-key on every request.
 *   OAuth:    Container CLI exchanges its placeholder token for a temp
 *             API key via /api/oauth/claude_cli/create_api_key.
 *             Proxy injects real OAuth token on that exchange request;
 *             subsequent requests carry the temp key which is valid as-is.
 */
import { createServer, Server } from 'http';
import { request as httpsRequest } from 'https';
import { request as httpRequest, RequestOptions } from 'http';

import { recordTokenUsage } from './db.js';
import { readEnvFile } from './env.js';
import { logger } from './logger.js';

export type AuthMode = 'api-key' | 'oauth';

export interface ProxyConfig {
  authMode: AuthMode;
}

export function startCredentialProxy(
  port: number,
  host = '127.0.0.1',
): Promise<Server> {
  const secrets = readEnvFile([
    'ANTHROPIC_API_KEY',
    'CLAUDE_CODE_OAUTH_TOKEN',
    'ANTHROPIC_AUTH_TOKEN',
    'ANTHROPIC_BASE_URL',
  ]);

  const authMode: AuthMode = secrets.ANTHROPIC_API_KEY ? 'api-key' : 'oauth';
  const oauthToken =
    secrets.CLAUDE_CODE_OAUTH_TOKEN || secrets.ANTHROPIC_AUTH_TOKEN;

  const upstreamUrl = new URL(
    secrets.ANTHROPIC_BASE_URL || 'https://api.anthropic.com',
  );
  const isHttps = upstreamUrl.protocol === 'https:';
  const makeRequest = isHttps ? httpsRequest : httpRequest;

  return new Promise((resolve, reject) => {
    const server = createServer((req, res) => {
      // Extract attribution query params (container, group) from the URL
      // and strip them before forwarding upstream
      const parsedUrl = new URL(req.url || '/', `http://${req.headers.host}`);
      const containerName = parsedUrl.searchParams.get('container') || 'unknown';
      const groupFolder = parsedUrl.searchParams.get('group') || 'unknown';
      parsedUrl.searchParams.delete('container');
      parsedUrl.searchParams.delete('group');
      const upstreamPath =
        parsedUrl.pathname +
        (parsedUrl.searchParams.size > 0
          ? `?${parsedUrl.searchParams.toString()}`
          : '');

      const chunks: Buffer[] = [];
      req.on('data', (c) => chunks.push(c));
      req.on('end', () => {
        const body = Buffer.concat(chunks);
        const headers: Record<string, string | number | string[] | undefined> =
          {
            ...(req.headers as Record<string, string>),
            host: upstreamUrl.host,
            'content-length': body.length,
          };

        // Strip hop-by-hop headers that must not be forwarded by proxies
        delete headers['connection'];
        delete headers['keep-alive'];
        delete headers['transfer-encoding'];

        if (authMode === 'api-key') {
          // API key mode: inject x-api-key on every request
          delete headers['x-api-key'];
          headers['x-api-key'] = secrets.ANTHROPIC_API_KEY;
        } else {
          // OAuth mode: replace placeholder Bearer token with the real one
          // only when the container actually sends an Authorization header
          // (exchange request + auth probes). Post-exchange requests use
          // x-api-key only, so they pass through without token injection.
          if (headers['authorization']) {
            delete headers['authorization'];
            if (oauthToken) {
              headers['authorization'] = `Bearer ${oauthToken}`;
            }
          }
        }

        const upstream = makeRequest(
          {
            hostname: upstreamUrl.hostname,
            port: upstreamUrl.port || (isHttps ? 443 : 80),
            path: upstreamPath,
            method: req.method,
            headers,
          } as RequestOptions,
          (upRes) => {
            const contentType = upRes.headers['content-type'] || '';
            const isSSE = contentType.includes('text/event-stream');
            const isJSON = contentType.includes('application/json');

            if (isSSE) {
              // Stream through immediately (zero latency), accumulate side buffer for token extraction
              res.writeHead(upRes.statusCode!, upRes.headers);
              const sseChunks: Buffer[] = [];
              upRes.on('data', (chunk: Buffer) => {
                sseChunks.push(chunk);
                res.write(chunk);
              });
              upRes.on('end', () => {
                res.end();
                try {
                  extractTokensFromSSE(
                    Buffer.concat(sseChunks).toString(),
                    containerName,
                    groupFolder,
                  );
                } catch {
                  // Never let token tracking break the proxy
                }
              });
            } else if (isJSON) {
              // Buffer full JSON response, extract tokens, then forward
              const jsonChunks: Buffer[] = [];
              upRes.on('data', (c: Buffer) => jsonChunks.push(c));
              upRes.on('end', () => {
                const responseBody = Buffer.concat(jsonChunks);
                res.writeHead(upRes.statusCode!, upRes.headers);
                res.end(responseBody);
                try {
                  extractTokensFromJSON(
                    responseBody.toString(),
                    containerName,
                    groupFolder,
                  );
                } catch {
                  // Never let token tracking break the proxy
                }
              });
            } else {
              // Other content types: pipe straight through
              res.writeHead(upRes.statusCode!, upRes.headers);
              upRes.pipe(res);
            }
          },
        );

        upstream.on('error', (err) => {
          logger.error(
            { err, url: req.url },
            'Credential proxy upstream error',
          );
          if (!res.headersSent) {
            res.writeHead(502);
            res.end('Bad Gateway');
          }
        });

        upstream.write(body);
        upstream.end();
      });
    });

    server.listen(port, host, () => {
      logger.info({ port, host, authMode }, 'Credential proxy started');
      resolve(server);
    });

    server.on('error', reject);
  });
}

/** Detect which auth mode the host is configured for. */
export function detectAuthMode(): AuthMode {
  const secrets = readEnvFile(['ANTHROPIC_API_KEY']);
  return secrets.ANTHROPIC_API_KEY ? 'api-key' : 'oauth';
}

function extractTokensFromJSON(
  body: string,
  containerName: string,
  groupFolder: string,
): void {
  const parsed = JSON.parse(body);
  if (!parsed.usage) return;
  recordTokenUsage({
    container_name: containerName,
    group_folder: groupFolder,
    model: parsed.model || null,
    input_tokens: parsed.usage.input_tokens || 0,
    output_tokens: parsed.usage.output_tokens || 0,
    cache_read_input_tokens: parsed.usage.cache_read_input_tokens || 0,
    cache_creation_input_tokens: parsed.usage.cache_creation_input_tokens || 0,
  });
}

function extractTokensFromSSE(
  data: string,
  containerName: string,
  groupFolder: string,
): void {
  // Claude API streaming: message_start has input token counts,
  // message_delta has output token counts. Collect both.
  let model: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  let cacheReadTokens = 0;
  let cacheCreationTokens = 0;

  for (const line of data.split('\n')) {
    if (!line.startsWith('data: ')) continue;
    const jsonStr = line.slice(6);
    if (jsonStr === '[DONE]') continue;
    try {
      const event = JSON.parse(jsonStr);
      if (event.type === 'message_start' && event.message) {
        model = event.message.model || model;
        if (event.message.usage) {
          inputTokens = event.message.usage.input_tokens || 0;
          cacheReadTokens = event.message.usage.cache_read_input_tokens || 0;
          cacheCreationTokens =
            event.message.usage.cache_creation_input_tokens || 0;
        }
      }
      if (event.type === 'message_delta' && event.usage) {
        outputTokens = event.usage.output_tokens || 0;
      }
    } catch {
      // Skip unparseable lines
    }
  }

  if (inputTokens > 0 || outputTokens > 0) {
    recordTokenUsage({
      container_name: containerName,
      group_folder: groupFolder,
      model,
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      cache_read_input_tokens: cacheReadTokens,
      cache_creation_input_tokens: cacheCreationTokens,
    });
  }
}
