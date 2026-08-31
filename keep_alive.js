const { chromium } = require('playwright');

const DEFAULT_ALLOWED_HOST = 'toxiscan.streamlit.app';
const APP_HEADING = 'ToxiScan';
const FAILURE_SCREENSHOT = 'keep_alive_screenshot.png';
const INITIAL_STATE_TIMEOUT_MS = 120000;
const WAKE_TIMEOUT_MS = 300000;
const CONNECTION_HOLD_MS = 15000;
const POLL_INTERVAL_MS = 1000;
const WAKE_CONTROL_PATTERN = /wake(?: up)?|back up|restart(?: this)? app|run(?: this)? app/i;

function validatedTarget(rawValue) {
  if (!rawValue) {
    throw new Error('A target URL is required.');
  }

  const target = new URL(rawValue);
  if (target.protocol !== 'https:') {
    throw new Error('The availability check requires HTTPS.');
  }

  const allowedHost = process.env.KEEP_ALIVE_ALLOWED_HOST || DEFAULT_ALLOWED_HOST;
  if (target.hostname !== allowedHost) {
    throw new Error(`The target host must be ${allowedHost}.`);
  }

  target.username = '';
  target.password = '';
  target.hash = '';
  return target.toString();
}

async function firstVisible(page, locatorFactory) {
  for (const frame of page.frames()) {
    const locator = locatorFactory(frame).first();
    if (await locator.isVisible().catch(() => false)) {
      return locator;
    }
  }
  return null;
}

async function applicationIsReady(page) {
  for (const frame of page.frames()) {
    const heading = frame.getByRole('heading').filter({ hasText: APP_HEADING }).first();
    const appContainer = frame.locator('[data-testid="stAppViewContainer"]').first();
    const headingVisible = await heading.isVisible().catch(() => false);
    const containerVisible = await appContainer.isVisible().catch(() => false);
    if (headingVisible && containerVisible) {
      return true;
    }
  }
  return false;
}

async function visibleWakeControl(page) {
  const button = await firstVisible(page, (frame) => frame.getByRole('button', {
    name: WAKE_CONTROL_PATTERN,
  }));
  if (button) {
    return button;
  }
  return firstVisible(page, (frame) => frame.getByRole('link', {
    name: WAKE_CONTROL_PATTERN,
  }));
}

async function observedFrames(page) {
  const observations = [];
  for (const frame of page.frames()) {
    const text = await frame.locator('body').innerText().catch(() => '');
    observations.push(`${frame.url()} :: ${text.replace(/\s+/g, ' ').trim().slice(0, 160) || '[no body text]'}`);
  }
  return observations.join(' | ');
}

async function waitForState(page, timeoutMs) {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    if (await applicationIsReady(page)) {
      return { state: 'ready' };
    }
    const wakeControl = await visibleWakeControl(page);
    if (wakeControl) {
      return { state: 'sleeping', wakeControl };
    }
    await page.waitForTimeout(POLL_INTERVAL_MS);
  }

  throw new Error(`Neither the application nor a Streamlit wake-up control became visible. Observed frames: ${await observedFrames(page)}`);
}

async function waitForApplication(page, timeoutMs) {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    if (await applicationIsReady(page)) {
      return;
    }
    await page.waitForTimeout(POLL_INTERVAL_MS);
  }

  throw new Error(`The application did not become ready after the wake-up request. Observed frames: ${await observedFrames(page)}`);
}

async function run() {
  let browser;
  let page;

  try {
    const target = validatedTarget(process.argv[2]);
    console.log('Starting the ToxiScan availability check.');

    browser = await chromium.launch({ headless: true });
    const context = await browser.newContext();
    page = await context.newPage();

    const response = await page.goto(target, {
      waitUntil: 'domcontentloaded',
      timeout: 90000,
    });
    if (response && response.status() >= 400) {
      throw new Error(`The app returned HTTP ${response.status()}.`);
    }

    const result = await waitForState(page, INITIAL_STATE_TIMEOUT_MS);
    if (result.state === 'sleeping') {
      console.log('The app is asleep; submitting the wake-up request.');
      await result.wakeControl.click({ timeout: 10000 });
      await waitForApplication(page, WAKE_TIMEOUT_MS);
    }

    await page.waitForTimeout(CONNECTION_HOLD_MS);
    if (!(await applicationIsReady(page))) {
      throw new Error('The application became unavailable during verification.');
    }

    console.log('The application interface is visible and responsive.');
  } catch (error) {
    console.error(`Availability check failed: ${error.message}`);
    process.exitCode = 1;

    if (page) {
      try {
        await page.screenshot({ path: FAILURE_SCREENSHOT, fullPage: true });
      } catch (screenshotError) {
        console.error(`Failure screenshot could not be saved: ${screenshotError.message}`);
      }
    }
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

run();
