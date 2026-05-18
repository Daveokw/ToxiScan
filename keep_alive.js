const { chromium } = require('playwright');

(async () => {
  const url = process.argv[2];
  if (!url) {
    console.error("Error: Please provide a URL as an argument.");
    process.exit(1);
  }

  console.log(`Starting keep-alive check for: ${url}`);
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
  });
  const page = await context.newPage();

  try {
    console.log("Navigating to the app URL...");
    // Streamlit apps can be slow to initialize, so we use a generous timeout
    await page.goto(url, { waitUntil: 'networkidle', timeout: 90000 });
  } catch (err) {
    console.log("Note: Page load timed out or encountered an error, checking for sleep indicators anyway:", err.message);
  }

  // Wait a bit to ensure the sleep screen has fully rendered if present
  await page.waitForTimeout(10000);

  try {
    // Check for the sleep page title or the "Yes, get this app back up!" button
    const wakeUpButton = page.locator('button:has-text("Yes, get this app back up!")');
    
    if (await wakeUpButton.count() > 0) {
      console.log("App is asleep! Clicking the 'Yes, get this app back up!' button...");
      await wakeUpButton.click();
      console.log("Clicked! Waiting 60 seconds for the app to wake up and spin up resources...");
      await page.waitForTimeout(60000);
      console.log("Wake up process triggered successfully!");
    } else {
      console.log("App is already awake! No sleep button was found.");
    }
  } catch (err) {
    console.error("An error occurred during wake-up checking:", err.message);
  } finally {
    try {
      await page.screenshot({ path: 'keep_alive_screenshot.png' });
      console.log("Screenshot saved as keep_alive_screenshot.png.");
    } catch (ssErr) {
      console.log("Could not take screenshot:", ssErr.message);
    }
    await browser.close();
    console.log("Keep-alive run finished successfully.");
  }
})();
