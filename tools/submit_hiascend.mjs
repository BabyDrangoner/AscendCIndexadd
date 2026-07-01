#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_CONTEST_ID = "41ffbad2024e4ccfa43520c57ffa7b9e";
const ORIGIN = "https://www.hiascend.com";
const API_BASE = `${ORIGIN}/ascendgateway/ascendservice`;
const STATIC_BASE = `${ORIGIN}/_static3`;
const SDK_FILES = ["CTAsUdWn.js", "B8KygROj.js", "JRuGLyjW.js", "l1gcMdIZ.js"];

function usage() {
  console.log(`Submit a CANN operator zip to the HiAscend contest through HTTP APIs.

Usage:
  ./tools/submit_hiascend.sh --zip /path/to/IndexAdd.zip --cookie 'name=value; ...'
  HIASCEND_COOKIE='name=value; ...' ./tools/submit_hiascend.sh --zip /path/to/IndexAdd.zip

Options:
  --zip FILE             Zip or run package to upload and submit
  --contest-id ID        Contest id (default: ${DEFAULT_CONTEST_ID})
  --topic TOPIC          Topic name when the contest stage requires one
  --cookie COOKIE        Raw Cookie header from a logged-in hiascend.com session
  --cookie-file FILE     File containing either a raw Cookie header or a "Cookie: ..." line
  --dry-run              Probe details and login-gated metadata, but do not upload/submit
  --help                 Show this help

Notes:
  The login cookie is required. The public contest detail API works without login,
  but submit metadata, temporary OBS credentials, upload, and submitWork require
  the user's authenticated hiascend.com session.
`);
}

function parseArgs(argv) {
  const args = {
    contestId: DEFAULT_CONTEST_ID,
    dryRun: false,
    cookie: process.env.HIASCEND_COOKIE || "",
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--help" || a === "-h") args.help = true;
    else if (a === "--dry-run") args.dryRun = true;
    else if (a === "--zip") args.zip = argv[++i];
    else if (a === "--contest-id") args.contestId = argv[++i];
    else if (a === "--topic") args.topic = argv[++i];
    else if (a === "--cookie") args.cookie = argv[++i];
    else if (a === "--cookie-file") args.cookieFile = argv[++i];
    else throw new Error(`unknown argument: ${a}`);
  }
  if (args.cookieFile) args.cookie = readCookieFile(args.cookieFile);
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

function ensureOk(data, context) {
  if (!data?.success) {
    const msg = data?.msg || data?.message || "request failed";
    const code = data?.code === undefined ? "" : ` code=${data.code}`;
    throw new Error(`${context} failed:${code} ${msg}`);
  }
  return data.data;
}

class HiascendClient {
  constructor({ cookie, contestId }) {
    this.cookieJar = parseCookieHeader(cookie);
    this.csrfToken = "";
    this.referer = `${ORIGIN}/developer/contests/details/${contestId}/submit`;
  }

  async request(method, apiPath, { query, body, retry406 = true } = {}) {
    const url = new URL(`${API_BASE}${apiPath}`);
    if (query) {
      for (const [k, v] of Object.entries(query)) {
        if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v));
      }
    }
    const headers = {
      accept: "application/json, text/plain, */*",
      locale: "zh-cn",
      origin: ORIGIN,
      referer: this.referer,
      "user-agent": "Mozilla/5.0 hiascend-api-submit/1.0",
    };
    const cookie = cookieHeader(this.cookieJar);
    if (cookie) headers.cookie = cookie;
    if (method !== "GET" && this.csrfToken) headers["x-csrf-token"] = this.csrfToken;
    const options = { method, headers, redirect: "manual" };
    if (body !== undefined) {
      headers["content-type"] = "application/json";
      options.body = JSON.stringify(body);
    }

    const res = await fetch(url, options);
    storeSetCookie(this.cookieJar, res.headers);
    const nextToken = res.headers.get("next-token");
    if (nextToken) this.csrfToken = nextToken;

    let data;
    const text = await res.text();
    try {
      data = JSON.parse(text);
    } catch {
      data = { success: false, code: res.status, msg: text.slice(0, 200) };
    }
    if (method !== "GET" && retry406 && Number(data?.code) === 406 && this.csrfToken) {
      return this.request(method, apiPath, { query, body, retry406: false });
    }
    return data;
  }

  get(apiPath, query) {
    return this.request("GET", apiPath, { query });
  }

  post(apiPath, body) {
    return this.request("POST", apiPath, { body });
  }
}

async function download(url, out) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed ${res.status}: ${url}`);
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, Buffer.from(await res.arrayBuffer()));
}

async function ensureObsSdk() {
  const dir = path.join(ROOT, ".cache", "hiascend_obs_sdk");
  fs.mkdirSync(dir, { recursive: true });
  await Promise.all(SDK_FILES.map(async (name) => {
    const out = path.join(dir, name);
    if (!fs.existsSync(out) || fs.statSync(out).size < 100) {
      await download(`${STATIC_BASE}/${name}`, out);
    }
  }));
  return path.join(dir, "CTAsUdWn.js");
}

function installBrowserPolyfills() {
  globalThis.window = globalThis;
  globalThis.addEventListener ||= () => {};
  globalThis.removeEventListener ||= () => {};
  globalThis.dispatchEvent ||= () => false;
  globalThis.__NUXT__ = { config: { app: { baseURL: "/", buildAssetsDir: "_static3", cdnURL: "" }, public: {} } };
  const element = () => ({
    style: {},
    setAttribute() {},
    appendChild() {},
    removeChild() {},
    remove() {},
    addEventListener() {},
    removeEventListener() {},
    getContext() { return null; },
  });
  globalThis.document = {
    documentElement: { style: {} },
    body: { appendChild() {}, removeChild() {} },
    head: { appendChild() {}, querySelector() { return null; } },
    createElement: element,
    addEventListener() {},
    removeEventListener() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
    getElementById() { return null; },
    cookie: "",
  };
  Object.defineProperty(globalThis, "navigator", {
    value: { userAgent: "node", maxTouchPoints: 0, platform: "MacIntel", language: "zh-CN" },
    configurable: true,
  });
  globalThis.location = { href: ORIGIN, origin: ORIGIN, host: "www.hiascend.com", pathname: "/" };
  globalThis.localStorage = { getItem() { return null; }, setItem() {}, removeItem() {} };
  globalThis.sessionStorage = globalThis.localStorage;
  globalThis.CSS = { supports() { return false; } };
  globalThis.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
  globalThis.MutationObserver = class { observe() {} disconnect() {} };
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
}

async function signObsPutUrl({ access, secret, securitytoken, server, bucket, key }) {
  installBrowserPolyfills();
  const sdkEntry = await ensureObsSdk();
  const { default: ObsClient } = await import(pathToFileURL(sdkEntry).href);
  const obs = new ObsClient({
    access_key_id: access,
    secret_access_key: secret,
    security_token: securitytoken,
    server,
    useRawXhr: true,
  });
  const result = obs.createSignedUrlSync({ Method: "PUT", Bucket: bucket, Key: key, Expires: 900 });
  if (!result?.SignedUrl) throw new Error("failed to create OBS signed URL");
  return result.SignedUrl;
}

async function uploadByObs(tempPass, zipPath) {
  const ext = path.basename(zipPath).split(".").pop();
  const key = `${tempPass.key}.${ext}`;
  const signedUrl = await signObsPutUrl({ ...tempPass, key });
  const bytes = fs.readFileSync(zipPath);
  const res = await fetch(signedUrl, { method: "PUT", body: bytes });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`OBS PUT failed: ${res.status} ${text.slice(0, 300)}`);
  }
  const requestId = res.headers.get("x-obs-request-id") || res.headers.get("x-amz-request-id");
  if (!requestId) throw new Error("OBS PUT succeeded but no request id header was returned");
  return { workId: requestId, key };
}

async function uploadWorkToObs(client, zipPath, { stageId, topic, privacyPolicy = 1 }, retry406 = true) {
  const bytes = fs.readFileSync(zipPath);
  const name = path.basename(zipPath);
  const form = new FormData();
  form.append("stageId", stageId);
  form.append("topic", topic || "");
  form.append("privacyPolicy", String(privacyPolicy));
  form.append("file", new Blob([bytes]), name);

  const url = `${API_BASE}/devCenter/contested/enrollment/uploadWorkToObs`;
  const headers = {
    accept: "application/json, text/plain, */*",
    locale: "zh-cn",
    origin: ORIGIN,
    referer: client.referer,
  };
  const cookie = cookieHeader(client.cookieJar);
  if (cookie) headers.cookie = cookie;
  if (client.csrfToken) headers["x-csrf-token"] = client.csrfToken;

  const res = await fetch(url, { method: "POST", headers, body: form });
  storeSetCookie(client.cookieJar, res.headers);
  const nextToken = res.headers.get("next-token");
  if (nextToken) client.csrfToken = nextToken;

  const text = await res.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { success: false, code: res.status, msg: text.slice(0, 200) };
  }
  if (retry406 && Number(data?.code) === 406 && client.csrfToken) {
    return uploadWorkToObs(client, zipPath, { stageId, topic, privacyPolicy }, false);
  }
  ensureOk(data, "uploadWorkToObs");
  return { workId: name };
}

async function uploadByCommons(client, zipPath) {
  const bytes = fs.readFileSync(zipPath);
  const name = path.basename(zipPath);
  const chunkSize = 5 * 1024 * 1024;
  const total = Math.max(1, Math.ceil(bytes.length / chunkSize));
  let fileId = "";
  for (let i = 0; i < total; i++) {
    const form = new FormData();
    form.append("file", new Blob([bytes.subarray(i * chunkSize, Math.min(bytes.length, (i + 1) * chunkSize))]), name);
    form.append("fileName", name);
    form.append("fileId", fileId);
    form.append("chunkNo", String(i + 1));
    form.append("chunkTotal", String(total));
    const url = `${API_BASE}/commons/upload/part`;
    const headers = {
      accept: "application/json, text/plain, */*",
      locale: "zh-cn",
      origin: ORIGIN,
      referer: client.referer,
    };
    const cookie = cookieHeader(client.cookieJar);
    if (cookie) headers.cookie = cookie;
    if (client.csrfToken) headers["x-csrf-token"] = client.csrfToken;
    const res = await fetch(url, { method: "POST", headers, body: form });
    storeSetCookie(client.cookieJar, res.headers);
    const nextToken = res.headers.get("next-token");
    if (nextToken) client.csrfToken = nextToken;
    const data = await res.json();
    const part = ensureOk(data, `upload chunk ${i + 1}/${total}`);
    fileId ||= part.fileId;
  }
  return { workId: fileId };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    usage();
    return;
  }
  if (!args.zip && !args.dryRun) throw new Error("--zip is required unless --dry-run is used");
  if (args.zip && !fs.existsSync(args.zip)) throw new Error(`zip not found: ${args.zip}`);

  const client = new HiascendClient({ cookie: args.cookie, contestId: args.contestId });
  const details = ensureOk(await client.get("/devCenter/contested/enrollment/details", { gameId: args.contestId }), "contest details");
  console.log(`contest: ${details.gameName}`);
  console.log(`stageId: ${details.stageId}`);
  console.log(`status: gameStatus=${details.gameStatus} auditFlag=${details.auditFlag} identity=${details.identity}`);

  const submitResp = await client.get("/devCenter/contested/enrollment/submitWorkDescription", { gameId: args.contestId });
  if (!submitResp?.success) {
    console.error(`submit metadata unavailable: code=${submitResp?.code} msg=${submitResp?.msg}`);
    if (Number(submitResp?.code) === 401) {
      console.error("login cookie is required. Pass --cookie or HIASCEND_COOKIE from a logged-in hiascend.com session.");
    }
    process.exit(args.dryRun ? 0 : 2);
  }
  const submitDetail = submitResp.data;
  const stageId = submitDetail.stageId || details.stageId;
  const topics = submitDetail.topics || [];
  const topic = args.topic ?? (topics.length === 1 ? topics[0] : "");
  if (topics.length > 1 && !topic) throw new Error(`topic is required; available topics: ${topics.join(", ")}`);
  console.log(`submit stageId: ${stageId}`);
  console.log(`obs upload: ${Boolean(submitDetail.obs)}`);
  if (topic) console.log(`topic: ${topic}`);
  if (args.dryRun) return;

  const stat = fs.statSync(args.zip);
  let upload;
  if (submitDetail.obs) {
    upload = await uploadWorkToObs(client, args.zip, {
      stageId,
      topic,
      privacyPolicy: 1,
    });
  } else {
    upload = await uploadByCommons(client, args.zip);
  }
  console.log(`uploaded workId: ${upload.workId}`);
  if (upload.key) console.log(`obs key: ${upload.key}`);

  const submitBody = {
    privacyPolicy: 1,
    stageId,
    topic,
    fileName: path.basename(args.zip),
    fileSize: stat.size,
    workId: upload.workId,
  };
  if (!topic) delete submitBody.topic;
  const submitted = await client.post("/devCenter/contested/enrollment/submitWork", submitBody);
  ensureOk(submitted, "submitWork");
  console.log("submitWork success");
  console.log(JSON.stringify(submitted, null, 2));
}

main().catch((err) => {
  console.error(err?.stack || err?.message || String(err));
  process.exit(1);
});
