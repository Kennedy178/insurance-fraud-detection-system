import sharp from "sharp";
import { readFileSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const SIZES = [72, 96, 128, 144, 152, 192, 384, 512];
const INPUT_SVG = join(__dirname, "public", "favicon.svg");
const OUTPUT_DIR = join(__dirname, "public", "icons");

mkdirSync(OUTPUT_DIR, { recursive: true });

const svgBuffer = readFileSync(INPUT_SVG);

async function generate() {
  for (const size of SIZES) {
    const outputPath = join(OUTPUT_DIR, `icon-${size}x${size}.png`);
    await sharp(svgBuffer).resize(size, size).png().toFile(outputPath);
    console.log(`Generated ${size}x${size}`);
  }

  await sharp(svgBuffer)
    .resize(512, 512)
    .png()
    .toFile(join(OUTPUT_DIR, "icon-512x512-maskable.png"));
  console.log("Generated 512x512-maskable");

  await sharp(svgBuffer)
    .resize(180, 180)
    .png()
    .toFile(join(OUTPUT_DIR, "apple-touch-icon.png"));
  console.log("Generated apple-touch-icon 180x180");

  console.log("All icons generated successfully.");
}

generate().catch(console.error);
