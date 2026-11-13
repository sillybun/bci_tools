import {test,expect} from '@playwright/test';
for (const compact of [false,true]) test('类别色块居中与默认配色 compact='+compact,async({page})=>{
 await page.goto('/');if(compact) await page.locator('#compactMode').check();
 const swatch=page.locator('[data-color="design"]'), name=page.locator('[data-category="design"] .category-label > div > strong');
 const a=await swatch.boundingBox(), b=await name.boundingBox();
 expect(Math.abs((a.y+a.height/2)-(b.y+b.height/2))).toBeLessThan(2);
 await swatch.click();
 await expect(page.locator('[data-palette="design"]')).toBeVisible();
 await expect(page.locator('[data-palette="design"] .color-swatch')).toHaveCount(8);
 await page.keyboard.press('Escape');await expect(swatch).toHaveAttribute('aria-expanded','false');await swatch.click();
 await page.locator('[data-palette-color="design"][data-value="#e06b5f"]').click();
 await expect(swatch).toHaveCSS('background-color','rgb(224, 107, 95)');
 await swatch.click();
 await page.locator('[data-custom-color="design"]').fill('#123456');
 await page.locator('[data-custom-color="design"]').dispatchEvent('change');
 await expect(swatch).toHaveCSS('background-color','rgb(18, 52, 86)');
});
