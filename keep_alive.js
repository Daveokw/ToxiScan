const { chromium } = require('playwright');

const DEFAULT_ALLOWED_HOST = 'toxiscan.streamlit.app';
const FAILURE_SCREENSHOT = 'keep_alive_screenshot.png';

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

    const wakeUpButton = page.getByRole('button', {
      name: 'Yes, get this app back up!',
      exact: true,
    });
    if (await wakeUpButton.count()) {
      await wakeUpButton.click({ timeout: 10000 });
      await page.waitForTimeout(15000);
      console.log('The app wake-up request was submitted.');
    } else {
      console.log('The app responded and was not showing the sleep control.');
    }
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
