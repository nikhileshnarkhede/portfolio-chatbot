# Make your own resume chatbot

This guide turns this repository into a chatbot that answers recruiters'
questions about **you**, in your own voice, using only what is in your resume.
You don't need to write any Python.

You need three things:

1. A GitHub account (free).
2. A Groq API key (free tier available) from <https://console.groq.com/keys>.
   It starts with `gsk_`.
3. Your resume as text, in the format described in [Step 2](#step-2-add-your-resume).

There are three ways to run it. Pick one.

| Option | Install anything? | Best for |
|---|---|---|
| [A. Streamlit Cloud](#option-a-streamlit-cloud-no-install) | No | A public link you can put on your resume or portfolio |
| [B. Your own computer](#option-b-your-own-computer-one-command) | Python 3.12 | Trying it out and editing privately |
| [C. Docker](#option-c-docker) | Docker | People who already use Docker |

> **Your resume becomes public if your repository is public.** Anything in
> `data/raw/resume.txt` can be read by anyone who opens your fork. Only put in
> what you are happy to share (for example, leave out your phone number).
> Your API key is different: it never goes in the repository.

---

## Step 1: Get your own copy

Click **Fork** at the top right of this repository's GitHub page, then
**Create fork**. You now have your own copy under your account.

(Optional: click **⭐ Star** too. It helps other people find the project.)

## Step 2: Add your resume

The chatbot answers from `data/raw/resume.txt`. Replace that file with your own.

1. Open [`data/raw/resume.example.txt`](../data/raw/resume.example.txt). It is a
   complete template for a made-up person, "Alex Rivera", with the five rules
   that make answers good written at the top.
2. Rewrite it with your own details. Keep the tags such as `<entry ...>` and
   `</entry>`; delete sections you don't need.
3. Save the result as `data/raw/resume.txt`, replacing the file that is there.

On GitHub you can do this in the browser: open `data/raw/resume.txt`, click the
pencil icon, paste your text over everything, and click **Commit changes**.

**Tip:** you can ask any AI assistant: *"Convert my resume below into the exact
format of this template"*, and paste both the template and your resume.

## Step 3: Add your name and links

Open [`configs/experiments/my_profile.json`](../configs/experiments/my_profile.json)
and replace every value that says `YOUR...`:

| Field | What to put |
|---|---|
| `assistant_label` | Your first name |
| `url_allowlist.urls` | Every link the bot is allowed to show. Copy them exactly as they appear in your resume. Any other link the AI writes is removed, so the bot can never invent a fake URL. |
| `ui.heading`, `ui.page_title` | For example `Chat with Alex` |
| `ui.tagline` | Your headline |
| `ui.assistant_avatar` | Any image URL. `https://github.com/<your-username>.png` is your GitHub photo. |
| `ui.contact` | The name, lines and links shown on the contact card. Keep your email as the last line. |

This is the only settings file you need. Everything else uses the defaults in
`configs/default.json`, which you can leave alone.

Then follow **one** of the options below.

---

## Option A: Streamlit Cloud (no install)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app** and choose to deploy from a GitHub repository.
3. Fill in:
   - **Repository:** `<your-username>/portfolio-chatbot` (your fork)
   - **Branch:** `master` (this repository's default branch)
   - **Main file path:** `ui/app.py` (not `app.py`)
4. Open **Advanced settings**:
   - **Python version:** 3.12
   - **Secrets:** paste these two lines, with your real key:

     ```toml
     GROQ_API_KEY = "gsk_your_key_here"
     EXPERIMENT = "my_profile"
     ```

5. Click **Deploy**. The first start takes a few minutes while it installs and
   builds the search index from your resume. After that it opens instantly.

You get a public link like `https://<something>.streamlit.app` to share.

To change the key later: your app → **⋮** → **Settings** → **Secrets**.

## Option B: Your own computer (one command)

Install [Python 3.12](https://www.python.org/downloads/) first. On Windows,
tick **"Add python.exe to PATH"** in the installer.

Download your fork: on its GitHub page click **Code → Download ZIP** and unzip
it, or, if you use git:

```bash
git clone https://github.com/<your-username>/portfolio-chatbot.git
cd portfolio-chatbot
```

Then run the setup script:

| System | Do this |
|---|---|
| Windows | Double-click `setup.bat` |
| macOS / Linux | Open a terminal in the folder and run `./setup.sh` |

The script creates a virtual environment, installs everything, asks for your
Groq key **once** (saved in `.streamlit/secrets.toml`, which git ignores), and
opens the chatbot at <http://localhost:8501>.

Run the same script again any time to start it again. After you edit your
resume, just restart; the index rebuilds automatically.

## Option C: Docker

```bash
cp .env.example .env
# edit .env: set GROQ_API_KEY=gsk_... and uncomment EXPERIMENT=my_profile
docker compose up --build
```

Open <http://localhost:8501>. Your resume is mounted from
`data/raw/resume.txt`, and the key comes from `.env` at run time. Neither is
baked into the image. More detail is in the [README's Docker section](../README.md#docker).

---

## Put it on your portfolio site

Add your Streamlit link to your resume and LinkedIn. To embed the chat in a
web page, use an iframe and add `?embed=true` to the URL:

```html
<iframe src="https://<your-app>.streamlit.app/?embed=true"
        width="100%" height="700" style="border:0"></iframe>
```

## FAQ

**Which API do I need, and what does it cost?**
A Groq API key from <https://console.groq.com/keys>. Groq has a free tier that
is usually enough for a personal resume bot; check
<https://groq.com/pricing> for current limits and prices. If you hit the free
limit, the bot automatically tries the next model in its list, and shows a
"try again later" message if all of them are busy.

**The bot still talks about the original author.**
Either `data/raw/resume.txt` is still the original file, or `EXPERIMENT =
"my_profile"` is missing from your secrets. Check both, then restart the app.

**The bot says it has no API key.**
On Streamlit Cloud, check **Settings → Secrets**. Locally, check that
`.streamlit/secrets.toml` exists and the key is inside quotes.

**A link I expected is missing from an answer.**
The link must appear in your resume *and* in `url_allowlist.urls` in
`configs/experiments/my_profile.json`, written exactly the same way.

**"ingest failed" or "Could not build the index".**
Your resume probably has a broken tag (for example `<entry ...>` without a
matching `</entry>`). Compare it with `data/raw/resume.example.txt`.

**Do I need the `eval/`, `tests/` and `scripts/` folders?**
No. They are the evaluation harness used to measure and tune the bot. The
chatbot runs without you touching them.

**Can I change the bot's personality?**
Yes, without Python. Edit `prompts/system/generic_persona.md`. Keep the
`{chat_history}`, `{context}` and `{question}` placeholders exactly as they are.

**Is my API key safe?**
It is only stored in Streamlit's Secrets or in `.streamlit/secrets.toml` /
`.env` on your machine. Those files are in `.gitignore`, so they are never
uploaded. Never paste your key into `resume.txt` or any `.json` file.

---

If this helped you, a ⭐ on the repository is the best thank-you.
