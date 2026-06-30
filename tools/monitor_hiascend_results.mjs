#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

const DEFAULT_CONTEST_ID = "41ffbad2024e4ccfa43520c57ffa7b9e";
const ORIGIN = "https://www.hiascend.com";
const API_BASE = `${ORIGIN}/ascendgateway/ascendservice`;

function usage() {
  console.log(`Monitor HiAscend contest submission and ranking result.

Usage:
  ./tools/monitor_hiascend_results.sh --topic IndexAdd --team 'TEAM_NAME'
  ./tools/monitor_hiascend_results.sh --once --topic IndexAdd --team 'TEAM_NAME'

Options:
  --topic TOPIC          Contest topic, for example IndexAdd
  --team TEAM            Team name used to filter ranking rows
  --contest-id ID        Contest id (default: ${DEFAULT_CONTEST_ID})
  --stage-id ID          Stage id; defaults to the value from contest details
  --cookie COOKIE        Raw Cookie header from a logged-in hiascend.com session
  --cookie-file FILE     File containing either a raw Cookie header or a "Cookie: ..." line
  --out-dir DIR          Output directory (default: .cache/hiascend_results)
  --interval SEC         Poll interval in seconds (default: 15)
  --page-size N          Ranking page size (default: 50)
  --once                 Probe once and exit
  --no-download          Do not download result txt files
  --help                 Show this help
`);
}

function parseArgs(argv) {
  const args = {
    contestId: DEFAULT_CONTEST_ID,
    cookie: process.env.HIASCEND_COOKIE || "",
    outDir: path.resolve(".cache", "hiascend_results"),
    interval: 15,
    pageSize: 50,
    once: false,
    download: true,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--help" || a === "-h") args.help = true;
    else if (a === "--once") args.once = true;
    else if (a === "--no-download") args.download = false;
    else if (a === "--topic") args.topic = argv[++i];
    else if (a === "--team") args.team = argv[++i];
    else if (a === "--contest-id") args.contestId = argv[++i];
    else if (a === "--stage-id") args.stageId = argv[++i];
    else if (a === "--cookie") args.cookie = argv[++i];
    else if (a === "--cookie-file") args.cookieFile = argv[++i];
    else if (a === "--out-dir") args.outDir = path.resolve(argv[++i]);
    else if (a === "--interval") args.interval = Number(argv[++i]);
    else if (a === "--page-size") args.pageSize = Number(argv[++i]);
    else throw new Error(`unknown argument: ${a}`);
  }
  if (args.cookieFile) args.cookie = readCookieFile(args.cookieFile);
  if (!args.help && !args.topic) throw new Error("--topic is required");
  if (!args.help && !args.cookie) throw new Error("--cookie or --cookie-file is required");
  if (!Number.isFinite(args.interval) || args.interval <= 0) throw new Error("--interval must be positive");
  if (!Number.isFinite(args.pageSize) || args.pageSize <= 0) throw new Error("--page-size must be positive");
  return args;
}

function readCookieFile(file) {
  const text = fs.readFileSync(file, "utf8").trim();
  const line = text.split(/\r?\n/).find((l) => /^cookie:/i.test(l)) || text;
  return line.replace(/^cookie:\s*/i, "").trim();
}

function parseCookieHeader(header) {
  const jar = new Map();
  for (const part of String(header || "").split(";")) {
    const idx = part.indexOf("=");
    if (idx > 0) jar.set(part.slice(0, idx).trim(), part.slice(idx + 1).trim());
  }
  return jar;
}

function cookieHeader(jar) {
  return [...jar.entries()].map(([k, v]) => `${k}=${v}`).join("; ");
}

function storeSetCookie(jar, headers) {
  const values = typeof headers.getSetCookie === "function"
    ? headers.getSetCookie()
    : splitSetCookie(headers.get("set-cookie"));
  for (const line of values || []) {
    const first = line.split(";")[0];
    const idx = first.indexOf("=");
    if (idx > 0) jar.set(first.slice(0, idx).trim(), first.slice(idx + 1).trim());
  }
}

function splitSetCookie(value) {
  if (!value) return [];
  return value.split(/,(?=[^;,]+=)/g);
}

class HiascendClient {
  constructor({ cookie, contestId }) {
    this.cookieJar = parseCookieHeader(cookie);
    this.referer = `${ORIGIN}/developer/contests/details/${contestId}`;
  }

  async get(apiPath, query = {}) {
    const url = new URL(`${API_BASE}${apiPath}`);
    for (const [k, v] of Object.entries(query)) {
      if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v));
    }
    const headers = {
      accept: "application/json, text/plain, */*",
      locale: "zh-cn",
      origin: ORIGIN,
      referer: this.referer,
      "user-agent": "Mozilla/5.0 hiascend-result-monitor/1.0",
    };
    const cookie = cookieHeader(this.cookieJar);
    if (cookie) headers.cookie = cookie;

    const res = await fetch(url, { headers, redirect: "manual" });
    storeSetCookie(this.cookieJar, res.headers);
    const text = await res.text();
    try {
      return JSON.parse(text);
    } catch {
      return { success: false, code: res.status, msg: text.slice(0, 300) };
    }
  }
}

function ensureSuccess(data, context) {
  if (!data?.success) {
    const msg = data?.msg || data?.message || "request failed";
    const code = data?.code === undefined ? "" : ` code=${data.code}`;
    throw new Error(`${context} failed:${code} ${msg}`);
  }
  return data.data;
}

function safeName(value) {
  return String(value || "unknown")
    .replace(/[\\/:"*?<>|]+/g, "_")
    .replace(/\s+/g, "_")
    .slice(0, 120);
}

function resultStamp(row) {
  const fromUrl = row?.url?.match(/_(\d{14})\.txt(?:$|\?)/)?.[1];
  if (fromUrl) return fromUrl;
  return String(row?.lastCommit || new Date().toISOString()).replace(/\D/g, "").slice(0, 14);
}

async function downloadText(url, out) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed: HTTP ${res.status}`);
  fs.writeFileSync(out, await res.text());
}

async function getMyRank(client, args) {
  const pageSize = Math.max(1, Math.min(100, Math.floor(args.pageSize)));
  for (let pageNo = 1; pageNo <= 20; pageNo++) {
    const data = ensureSuccess(await client.get(
      "/devCenter/contested/enrollment/getWorkPerformancesRankList",
      {
        gameId: args.contestId,
        teamName: args.team || "",
        pageNo,
        pageSize,
        topic: args.topic,
      },
    ), "rank list");
    const rows = data?.list || [];
    const row = args.team ? rows.find((r) => r.teamName === args.team) || rows[0] : rows[0];
    if (row || pageNo >= Number(data?.pages || 1)) return { page: data, row: row || null };
  }
  return { page: null, row: null };
}

async function pollOnce(client, args, state) {
  const details = ensureSuccess(await client.get(
    "/devCenter/contested/enrollment/details",
    { gameId: args.contestId },
  ), "contest details");
  const stageId = args.stageId || details.stageId;
  const workInfo = ensureSuccess(await client.get(
    "/devCenter/contested/enrollment/getWorkInfo",
    { stageId, topic: args.topic },
  ), "work info");
  const rank = await getMyRank(client, args);
  const row = rank.row;
  const now = new Date().toISOString();
  const snapshot = {
    ts: now,
    topic: args.topic,
    team: args.team || null,
    stageId,
    workInfo,
    rank: row,
  };

  fs.mkdirSync(args.outDir, { recursive: true });
  const prefix = safeName(args.topic);
  fs.writeFileSync(path.join(args.outDir, `${prefix}_latest.json`), `${JSON.stringify(snapshot, null, 2)}\n`);
  fs.appendFileSync(path.join(args.outDir, `${prefix}.ndjson`), `${JSON.stringify(snapshot)}\n`);

  let downloaded = "";
  if (args.download && row?.url) {
    const stamp = resultStamp(row);
    const out = path.join(args.outDir, `Results_${prefix}_${stamp}.txt`);
    if (!fs.existsSync(out)) {
      await downloadText(row.url, out);
      downloaded = out;
    }
  }

  const key = JSON.stringify({
    workName: workInfo?.workName,
    uploadTime: workInfo?.uploadTime,
    rankTime: row?.lastCommit,
    takeTime: row?.takeTime,
    url: row?.url,
  });
  const changed = key !== state.lastKey;
  state.lastKey = key;
  return { changed, snapshot, downloaded };
}

function formatLine(snapshot, downloaded) {
  const w = snapshot.workInfo || {};
  const r = snapshot.rank || {};
  const pieces = [
    `[${snapshot.ts}]`,
    `topic=${snapshot.topic}`,
    `work=${w.workName || "-"}`,
    `upload=${w.uploadTime || "-"}`,
    `rank=${r.ranking ?? "-"}`,
    `takeTime=${r.takeTime ?? "-"}us`,
    `lastCommit=${r.lastCommit || "-"}`,
    `commitTimes=${r.commitTimes ?? "-"}`,
  ];
  if (downloaded) pieces.push(`downloaded=${downloaded}`);
  else if (r.url) pieces.push("resultUrl=seen");
  return pieces.join(" ");
}

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    usage();
    return;
  }
  const client = new HiascendClient(args);
  const state = { lastKey: "" };
  while (true) {
    try {
      const result = await pollOnce(client, args, state);
      if (result.changed || args.once) {
        console.log(formatLine(result.snapshot, result.downloaded));
      }
    } catch (err) {
      console.error(`[${new Date().toISOString()}] ${err?.message || String(err)}`);
    }
    if (args.once) return;
    await sleep(args.interval * 1000);
  }
}

main().catch((err) => {
  console.error(err?.stack || err?.message || String(err));
  process.exit(1);
});
