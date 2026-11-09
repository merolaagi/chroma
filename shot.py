import asyncio, sys
from playwright.async_api import async_playwright
PAGES=["world","predict","voting","genes","modules","attractors","loop","hypercube"]
async def main():
    errs=[]
    async with async_playwright() as pw:
        b=await pw.chromium.launch(args=["--no-sandbox"])
        pg=await b.new_page(viewport={"width":1340,"height":950})
        pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type=="error" else None)
        pg.on("pageerror", lambda e: errs.append("pageerror: "+str(e)))
        await pg.goto("http://127.0.0.1:51999/", wait_until="networkidle")
        await pg.wait_for_timeout(900)
        for p in PAGES:
            await pg.click(f'a[data-page="{p}"]')
            await pg.wait_for_timeout(500)
            if p=="loop": await pg.click("text=Walk a closed path"); await pg.wait_for_timeout(600)
            await pg.screenshot(path=f"shot-{p}.png")
        # dark
        await pg.click('a[data-page="world"]'); await pg.click('[data-ink=theme]')
        await pg.wait_for_timeout(500); await pg.screenshot(path="shot-dark.png")
        # phone
        pg2=await b.new_page(viewport={"width":390,"height":844})
        await pg2.goto("http://127.0.0.1:51999/", wait_until="networkidle")
        await pg2.wait_for_timeout(800); await pg2.screenshot(path="shot-phone.png")
        await b.close()
    print("CONSOLE ERRORS:", len(errs))
    for e in errs[:12]: print(" ", e)
asyncio.run(main())
