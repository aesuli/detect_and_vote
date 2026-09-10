# Detect & Vote

A browser-based live voting app for public, interactive installations. It uses a camera feed plus object detection to let a group vote by either:

- moving into a marked region on the screen, or
- showing a specific object that matches an answer category.

The project is designed for public environments, classroom demos, interactive exhibits, and quick live quiz setups. It combines a real-time camera stream, region counting, question playback, and answer voting into a single web interface.

## What the app does

The web app runs a live video stream and overlays detection boxes and voting regions. Users can vote without a separate input device:

1. In region-based mode, you draw one or more answer zones on the screen. Each zone is assigned to an answer slot. When a detected object overlaps that zone, it contributes to that answer.
2. In object-list mode, you define which objects belong to Answer 1 and which belong to Answer 2. The app counts detected objects and assigns the vote to the matching answer.

This makes it well suited for public demos where people physically move, gesture, or show objects in front of the camera.

## Example use cases

- Ask a question such as: "Which answer is correct?"
- Show two answer choices on screen
- Let participants vote by standing in the correct area or by showing a matching object
- The app tracks live counts, updates the timer, and displays the final result after the vote window closes

The voting flow can also run a set of prepared quiz questions from JSONL/JSON/ZIP files, with a countdown, answer review, and visual result display.

## Running the app

Install the project dependencies:

```bash
pip install -r requirements.txt
```

If needed, install PyTorch for your system first, as the project depends on the Hugging Face detector stack. The repository includes a comment in the requirements file showing the standard pattern for installing a matching torch build.

Then launch the web app:

```bash
python web_detect_and_vote.py
```

By default, the app starts a CherryPy server on:

```text
http://localhost:8080
```

Open that URL in a browser. The page includes:

- the camera view with detection overlays
- the configuration panel
- the live vote count panel
- the result graph

## Command-line options

The app supports a few startup flags for the camera and detector setup:

```bash
python web_detect_and_vote.py \
  --detector owlv2 \
  --model-name "google/owlv2-base-patch16-ensemble" \
  --objects "a person" "human face" "a hand" \
  --threshold 0.17 \
  --frame-width 960 \
  --frame-height 540 \
  --video-device-id 0 \
  --host 127.0.0.1 \
  --port 8080
```

Common arguments:

- `--detector`: `owlv2` or `owlvit`
- `--model-name`: optional model override when you want a specific pretrained model
- `--objects`: objects to detect in the scene when region voting is used
- `--threshold`: detection confidence threshold
- `--frame-width` / `--frame-height`: camera resolution
- `--video-device-id`: webcam index, usually `0` for the default camera
- `--host`: host address to bind to (default: `127.0.0.1`)
- `--port`: port to listen on (default: `8080`)

## Configuring the app

The app exposes all the key settings in the browser UI. The configuration panel is the main place to tune the system for a public setup.

### 1. Vote mapping mode

Choose between two voting modes:

- `Regions choose answer`: draw regions on the frame; each region belongs to an answer slot
- `Object lists choose answer`: define object labels for each answer; a detected object decides the vote

This is the most important setting for public interaction.

### 2. Detector settings

In the Detection section you can configure:

- detector type: OWL-ViT or OWL-v2
- model name
- confidence threshold
- frame skip
- detected objects list

A typical setup for a public audience is to detect broad, easy-to-show labels such as:

```text
a person
human face
a hand
```

When using object-list voting, define the object lists for each answer separately. For example, Answer 1 can include `the palm of an open hand`, while Answer 2 can include `a hand closed in a fist`.

### 3. Region voting setup

For region-based voting:

- create one or more polygons on the video frame
- assign each region to Answer 1 or Answer 2
- choose overlap or inside matching criteria
- pick colors for the answer regions

This mode is ideal when you want people to physically stand or move into a particular area to vote.

### 4. Voting timing and display

The voting controls let you set:

- vote timer in seconds
- pause duration between questions
- pre-question countdown
- rolling window size for recent counts
- whether to shuffle answer order
- whether to add reading time based on text length

For public installations, a timer of 20–30 seconds is usually a good default. The countdown allows people to prepare before the question begins.

### 5. Question files

The app loads questions from the `data` directory by default. The bundled example uses `data/test.jsonl`.

The supported formats are:

- `.jsonl`
- `.json`
- `.zip`

Question files may contain one or more questions with answers and an optional correct answer. The app also supports uploaded question sources from the browser.

A typical question entry in JSONL looks like this:

```json
{"question":"Which planet is the largest in our solar system?","answers":["Jupiter","Mars"],"correct_answer":"Jupiter"}
```

You can also include richer content such as markdown, extra time, or image links, depending on the format expected by the app.

## Voting flow

The app supports a full timed quiz loop:

1. Show a pre-question countdown
2. Display the question and answer choices
3. Count votes from the current camera scene during the vote window
4. Pause briefly to show the result
5. Continue to the next question

During the vote period, the UI updates live counts and graph history. After the voting window closes, it shows the winning answer and can compare the result to the correct answer when one is provided.

## Public installation tips

For a public environment, a few practical choices make the experience more reliable:

- Use stable lighting and avoid strong glare or shadows
- Keep the camera at a consistent distance from the audience
- Use broad object labels that are easy to trigger
- Keep the threshold at a safe value for the room
- Use large regions for audience voting if people are standing in fixed positions
- Run the app on a laptop or mini PC connected to a projector or screen

## Stopping the app

Press `Ctrl+C` in the terminal to shut the server down cleanly.

## Project files

- `web_detect_and_vote.py`: CherryPy web app and voting logic
- `templates/web_detect_and_vote.html`: browser interface
- `static/js/web_detect_and_vote.js`: live dashboard and interaction logic
- `static/css/web_detect_and_vote.css`: app styling
- `data/`: question datasets and examples

## License

See [license file](LICENSE).
