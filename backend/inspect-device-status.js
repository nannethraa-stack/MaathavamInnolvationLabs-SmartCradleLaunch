const db = require("better-sqlite3")("./cradle.db");

const rows = db.prepare(`
  SELECT rowid, *
  FROM device_status
  ORDER BY rowid DESC
  LIMIT 10
`).all();

console.log(JSON.stringify(rows, null, 2));
