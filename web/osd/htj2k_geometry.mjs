// Pure geometry mapping OpenSeadragon (level, x, y) tile coordinates onto
// HTJ2K windowed region-decode requests.  No browser / Worker / WASM
// dependencies, so it is shared verbatim by the decode core, the Web Worker,
// the OSD TileSource and the Node correctness test.
//
// Model (matches OpenSeadragon.TileSource defaults exactly):
//   maxLevel = ceil(log2(max(fullW, fullH)))   -- the full-resolution level
//   reduce   = maxLevel - level                -- HTJ2K resolution reduction
//   levelW   = ceil(fullW / 2^reduce)          -- decoded width at that level
//   tile (x,y) covers [x*ts, min((x+1)*ts, levelW)) x [y*ts, min(.., levelH))
//
// Only reduce in [0, maxReduce] is decodable (maxReduce = the codestream's DWT
// decomposition levels), so minLevel = maxLevel - maxReduce: OSD never requests
// a coarser level than the pyramid can produce.

// ceil(log2(n)) computed the same way OpenSeadragon does (Math.log/Math.LN2,
// not Math.log2) so our maxLevel byte-matches OSD's when it derives its own.
function ceilLog2(n) {
  return Math.ceil(Math.log(n) / Math.LN2);
}

export function computePyramid({ fullW, fullH, tileSize = 256, maxReduce }) {
  const maxLevel = ceilLog2(Math.max(fullW, fullH));
  const minLevel = Math.max(0, maxLevel - maxReduce);
  return { fullW, fullH, tileSize, maxReduce, maxLevel, minLevel };
}

export function levelDims(level, p) {
  const reduce = p.maxLevel - level;
  const s = 1 << reduce;
  // For integer dims and power-of-two s, Math.ceil(W/s) is exact and equals the
  // decoder's (W + s - 1) >> reduce reduced-level width.
  return { reduce, levelW: Math.ceil(p.fullW / s), levelH: Math.ceil(p.fullH / s) };
}

// Map an OSD tile (level, tx, ty) to a clipped HTJ2K region-decode request.
// Returns null if the tile is out of the decodable range or off the image.
export function tileRegion(level, tx, ty, p) {
  const reduce = p.maxLevel - level;
  if (reduce < 0 || reduce > p.maxReduce) return null;
  const { levelW, levelH } = levelDims(level, p);
  const ts = p.tileSize;
  const x0 = tx * ts;
  const y0 = ty * ts;
  if (x0 >= levelW || y0 >= levelH) return null;
  const w = Math.min(ts, levelW - x0);
  const h = Math.min(ts, levelH - y0);
  return { reduce, x0, y0, w, h, levelW, levelH };
}

// Number of tiles at a level (mirrors OSD getNumTiles for tileOverlap = 0).
export function numTiles(level, p) {
  const { levelW, levelH } = levelDims(level, p);
  return { nx: Math.ceil(levelW / p.tileSize), ny: Math.ceil(levelH / p.tileSize) };
}
