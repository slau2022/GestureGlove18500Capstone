int x;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);
}

void loop() {
  while (!Serial.available());
  String read_in = Serial.readString();
  if (read_in == "q")
    exit(0);
  else
    x = read_in.toInt();
  Serial.print(x + 1);
}
