const { chromium } = require('playwright');

const DEFAULT_ALLOWED_HOST = 'toxiscan.streamlit.app';
const APP_HEADING = 'ToxiScan';
const FAILURE_SCREENSHOT = 'keep_alive_screenshot.png';
const INITIAL_STATE_TIMEOUT_MS = 120000;
const WAKE_TIMEOUT_MS = 300000;
const CONNECTION_HOLD_MS = 15000;

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

async function waitForState(page, timeoutMs) {
  const appHeading = page.getByRole('heading').filter({ hasText: APP_HEADING }).first();
  const wakeUpButton = page.getByRole('button', {
    name: /get this app back up/i,
  }).first();
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    if (await appHeading.isVisible()) {
      return { state: 'ready', appHeading, wakeUpButton };
    }
    if (await wakeUpButton.isVisible()) {
      return { state: 'sleeping', appHeading, wakeUpButton };
    }
    await page.waitForTimeout(1000);
  }

  throw new Error('Neither the application nor the Streamlit wake-up control became visible.');
}

async function waitForApplication(appHeading, page, timeoutMs) {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    if (await appHeading.isVisible()) {
      return;
    }
    await page.waitForTimeout(1000);
  }

  throw new Error('The application did not become ready after the wake-up request.');
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
      await result.wakeUpButton.click({ timeout: 10000 });
      await waitForApplication(result.appHeading, page, WAKE_TIMEOUT_MS);
    }

    await page.waitForTimeout(CONNECTION_HOLD_MS);
    if (!(await result.appHeading.isVisible())) {
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
