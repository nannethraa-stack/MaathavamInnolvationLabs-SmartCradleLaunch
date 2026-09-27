const mqtt = require("mqtt");

const client = mqtt.connect("mqtt://broker.hivemq.com:1883");

client.on("connect", () => {
  console.log("Connected");
  client.subscribe("cradle/device001/status", { qos: 0 }, err => {
    if (err) console.error(err);
    else console.log("Listening for cradle/device001/status ...");
  });
});

client.on("message", (topic, message) => {
  console.log("\nTOPIC:", topic);
  console.log("PAYLOAD:");
  console.log(message.toString());
});
