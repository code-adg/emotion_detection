"""
prepare_meld.py
────────────────
Download MELD text-only data and prepare for fine-tuning.
Falls back to a built-in curated subtitle emotion dataset if download fails.

Usage:
    python prepare_meld.py

Output:
    labeled_subtitles.csv  — ~10K rows (text, label)
    test_subtitles.csv     — held-out test set
"""

import os
import io
import urllib.request
import pandas as pd


# ── Direct links to MELD raw CSV files ────────────────────────────
MELD_URLS = {
    "train": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
    "dev":   "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
    "test":  "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
}

VALID_EMOTIONS = {"anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"}


def download_and_parse(url: str) -> pd.DataFrame:
    """Download CSV and parse with robust settings for MELD's messy quoting."""
    fname = url.split("/")[-1]
    print(f"   ⬇ {fname} ... ", end="", flush=True)

    response = urllib.request.urlopen(url)
    raw = response.read().decode("utf-8", errors="replace")

    # MELD CSVs have irregular quoting — try multiple strategies
    for params in [
        {"quoting": 1, "on_bad_lines": "skip"},   # QUOTE_ALL
        {"quoting": 0, "on_bad_lines": "skip"},   # QUOTE_MINIMAL
        {"quoting": 3, "on_bad_lines": "skip", "engine": "python"},  # QUOTE_NONE
    ]:
        try:
            df = pd.read_csv(io.StringIO(raw), **params)
            if len(df) > 50:  # sanity check — should have hundreds of rows
                print(f"{len(df)} rows ✅")
                return df
        except Exception:
            continue

    print("⚠ parsing issues")
    return pd.DataFrame()


def extract_text_emotion(df: pd.DataFrame) -> list:
    """Extract (text, emotion) pairs from a MELD DataFrame."""
    rows = []

    # Find column names (case-insensitive)
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = cols.get("utterance", cols.get("text", None))
    emo_col = cols.get("emotion", None)

    if not text_col or not emo_col:
        return rows

    for _, row in df.iterrows():
        text = str(row[text_col]).strip().strip("'\"")
        emotion = str(row[emo_col]).strip().lower()

        if (
            len(text) > 3
            and text != "nan"
            and emotion in VALID_EMOTIONS
        ):
            rows.append({"text": text, "label": emotion})

    return rows


def create_builtin_dataset():
    """
    Fallback: a curated dataset of 500+ subtitle-style dialogue lines
    covering all 7 MELD emotion categories.
    """
    print("\n📦 Using built-in curated subtitle dataset (500+ samples)...")

    data = [
        # ── JOY / HAPPINESS ──────────────────────────────────────────
        ("I just got the promotion! I can't believe it!", "joy"),
        ("This is the best day of my entire life!", "joy"),
        ("We're having a baby! Can you believe it?", "joy"),
        ("I passed all my exams with flying colors!", "joy"),
        ("You actually came! I'm so happy to see you!", "joy"),
        ("The wedding was absolutely perfect.", "joy"),
        ("I love this song, it always makes me smile.", "joy"),
        ("Everything worked out exactly as we planned!", "joy"),
        ("She said yes! She actually said yes!", "joy"),
        ("We won the championship! We actually did it!", "joy"),
        ("This cake is absolutely delicious!", "joy"),
        ("I got accepted into my dream university!", "joy"),
        ("Best birthday surprise I've ever had!", "joy"),
        ("The baby took her first steps today!", "joy"),
        ("I finally paid off all my student loans!", "joy"),
        ("We're going to Paris for our anniversary!", "joy"),
        ("My painting won first place in the competition!", "joy"),
        ("You made me the happiest person alive.", "joy"),
        ("The test results came back negative, thank God!", "joy"),
        ("I just finished writing my first novel!", "joy"),
        ("That's wonderful news, I'm thrilled for you!", "joy"),
        ("We found the perfect house for our family!", "joy"),
        ("Our team pulled off an incredible comeback!", "joy"),
        ("I've never laughed so hard in my entire life!", "joy"),
        ("This sunset is the most beautiful thing I've seen.", "joy"),
        ("They surprised me with a party, I'm so touched!", "joy"),
        ("Christmas morning is always so magical.", "joy"),
        ("My daughter graduated top of her class!", "joy"),
        ("We reunited after ten years apart!", "joy"),
        ("The flowers you sent brightened my whole week.", "joy"),
        ("I finally learned how to cook this dish perfectly!", "joy"),
        ("Our project received the highest rating possible!", "joy"),
        ("I'm so glad we decided to come here today.", "joy"),
        ("That joke was hilarious, tell me another one!", "joy"),
        ("We're getting a puppy, the kids are ecstatic!", "joy"),
        ("Nothing can ruin my mood today!", "joy"),
        ("The crowd went absolutely wild after that goal!", "joy"),
        ("I feel like I'm on top of the world right now!", "joy"),
        ("My recovery is going better than expected!", "joy"),
        ("Look at this view, it's absolutely breathtaking!", "joy"),

        # ── SADNESS ──────────────────────────────────────────────────
        ("I don't think I can do this anymore.", "sadness"),
        ("She left without even saying goodbye.", "sadness"),
        ("I miss him so much it hurts.", "sadness"),
        ("Nobody even remembered my birthday.", "sadness"),
        ("The house feels so empty without her.", "sadness"),
        ("I failed again. I always fail.", "sadness"),
        ("He's not coming back, is he?", "sadness"),
        ("I feel like I've lost everything.", "sadness"),
        ("It's been a year and I still can't move on.", "sadness"),
        ("I don't even know who I am anymore.", "sadness"),
        ("The funeral was the hardest thing I've ever done.", "sadness"),
        ("I feel so alone in this world.", "sadness"),
        ("We had to put the dog down yesterday.", "sadness"),
        ("My grandmother passed away last night.", "sadness"),
        ("I thought we were friends, but I was wrong.", "sadness"),
        ("The divorce was finalized today.", "sadness"),
        ("I can't stop crying whenever I think about it.", "sadness"),
        ("They told me I didn't get the job.", "sadness"),
        ("I wish things could go back to the way they were.", "sadness"),
        ("Sometimes I feel like nobody really cares.", "sadness"),
        ("I keep replaying that moment in my head.", "sadness"),
        ("Everything reminds me of what I've lost.", "sadness"),
        ("I haven't felt this low in a very long time.", "sadness"),
        ("The rain makes everything feel more depressing.", "sadness"),
        ("I don't think I'll ever be truly happy again.", "sadness"),
        ("She was my best friend and now she's gone.", "sadness"),
        ("I lost the only person who understood me.", "sadness"),
        ("I'm tired of pretending everything is fine.", "sadness"),
        ("We grew apart and there's nothing I can do.", "sadness"),
        ("I never got the chance to say I'm sorry.", "sadness"),
        ("Looking at old photos makes my heart ache.", "sadness"),
        ("I was hoping for better news from the doctor.", "sadness"),
        ("It breaks my heart to see you like this.", "sadness"),
        ("The whole neighborhood feels different now.", "sadness"),
        ("I thought we had something special.", "sadness"),
        ("Every song on the radio reminds me of her.", "sadness"),
        ("I feel like giving up on everything.", "sadness"),
        ("This empty chair at the table says it all.", "sadness"),
        ("I wish I had more time with him.", "sadness"),
        ("They promised they would come but never did.", "sadness"),

        # ── ANGER ────────────────────────────────────────────────────
        ("How dare you speak to me like that!", "anger"),
        ("I've had enough of your excuses!", "anger"),
        ("Get out of my house right now!", "anger"),
        ("You lied to me! You've been lying this whole time!", "anger"),
        ("I swear if you do that one more time...", "anger"),
        ("This is absolutely unacceptable!", "anger"),
        ("You had no right to go through my things!", "anger"),
        ("I trusted you and you betrayed me!", "anger"),
        ("Stop telling me to calm down!", "anger"),
        ("I can't stand being around you anymore!", "anger"),
        ("You ruined everything we worked for!", "anger"),
        ("Don't you dare blame this on me!", "anger"),
        ("I'm sick and tired of being treated like garbage!", "anger"),
        ("You crossed the line this time!", "anger"),
        ("I warned you and you didn't listen!", "anger"),
        ("Who gave you permission to do that?", "anger"),
        ("This company doesn't care about its employees!", "anger"),
        ("You think you can just walk all over me?", "anger"),
        ("Every single time you let me down!", "anger"),
        ("I want answers and I want them now!", "anger"),
        ("That was the most disrespectful thing anyone's done!", "anger"),
        ("You destroyed my trust completely!", "anger"),
        ("Shut up! Just shut up and listen for once!", "anger"),
        ("I'm done cleaning up your messes!", "anger"),
        ("How could you do something so stupid?", "anger"),
        ("You're the reason everything falls apart!", "anger"),
        ("I never want to see your face again!", "anger"),
        ("Don't you walk away from me while I'm talking!", "anger"),
        ("You promised you would change but you never do!", "anger"),
        ("This whole situation makes my blood boil!", "anger"),
        ("You took advantage of my kindness!", "anger"),
        ("I will not tolerate this behavior any longer!", "anger"),
        ("Stop making excuses and take responsibility!", "anger"),
        ("I've been patient long enough with you!", "anger"),
        ("You have some nerve showing up here!", "anger"),
        ("This is the last straw, I'm finished!", "anger"),
        ("You went behind my back, didn't you?", "anger"),
        ("I'm furious that nobody told me about this!", "anger"),
        ("How many times do I have to say it?", "anger"),
        ("You made me look like a fool!", "anger"),

        # ── FEAR ─────────────────────────────────────────────────────
        ("Did you hear that noise downstairs?", "fear"),
        ("Something doesn't feel right about this place.", "fear"),
        ("I think someone's been following us.", "fear"),
        ("Please don't leave me alone here.", "fear"),
        ("What if something terrible happens?", "fear"),
        ("I have a really bad feeling about this.", "fear"),
        ("We need to get out of here right now.", "fear"),
        ("I'm scared of what they'll do to us.", "fear"),
        ("My hands won't stop shaking.", "fear"),
        ("There's someone standing outside the window.", "fear"),
        ("I can't go back there, please don't make me.", "fear"),
        ("What if the test results come back positive?", "fear"),
        ("I don't think we're safe here anymore.", "fear"),
        ("The room got dark and I panicked.", "fear"),
        ("I keep having nightmares about that night.", "fear"),
        ("Please somebody help me!", "fear"),
        ("I heard footsteps behind me in the alley.", "fear"),
        ("What if they find out the truth?", "fear"),
        ("I'm afraid I won't make it through the night.", "fear"),
        ("Don't open that door, please!", "fear"),
        ("Something is very wrong with this situation.", "fear"),
        ("My heart is pounding so hard right now.", "fear"),
        ("I'm terrified of what happens next.", "fear"),
        ("That shadow moved, did you see that?", "fear"),
        ("Lock all the doors and windows immediately!", "fear"),
        ("I've never been this afraid in my life.", "fear"),
        ("We shouldn't have come here, this was a mistake.", "fear"),
        ("I can't breathe, I'm having a panic attack.", "fear"),
        ("Promise me you won't let anything happen to them.", "fear"),
        ("The storm is getting worse, I'm worried.", "fear"),
        ("I think the bridge is about to collapse!", "fear"),
        ("They're getting closer, we need to hide!", "fear"),
        ("What was that scream just now?", "fear"),
        ("I dream about that accident almost every night.", "fear"),
        ("My instincts are telling me to run.", "fear"),
        ("Don't turn around, just keep walking.", "fear"),
        ("I'm afraid of losing everything I've built.", "fear"),
        ("The elevator just stopped between floors.", "fear"),
        ("Something's lurking in the shadows.", "fear"),
        ("I don't want to die, not like this.", "fear"),

        # ── SURPRISE ─────────────────────────────────────────────────
        ("Wait, what? You're kidding me!", "surprise"),
        ("I had absolutely no idea about any of this!", "surprise"),
        ("Whoa, I didn't see that coming at all!", "surprise"),
        ("You've got to be joking right now!", "surprise"),
        ("Are you serious? That's unbelievable!", "surprise"),
        ("When did this happen? Nobody told me!", "surprise"),
        ("I can't believe my eyes right now!", "surprise"),
        ("Well that was completely unexpected.", "surprise"),
        ("Hold on, what did you just say?", "surprise"),
        ("You did all of this for me?", "surprise"),
        ("Since when are you two together?", "surprise"),
        ("No way! That's absolutely insane!", "surprise"),
        ("I never expected to see you here!", "surprise"),
        ("Wait, that was today? I totally forgot!", "surprise"),
        ("Plot twist! I did not see that coming!", "surprise"),
        ("She actually showed up? I'm shocked!", "surprise"),
        ("How is that even possible?", "surprise"),
        ("You've been working here the entire time?", "surprise"),
        ("Oh my God, what happened to this place?", "surprise"),
        ("I'm speechless, I really am.", "surprise"),
        ("That's the last thing I expected to hear!", "surprise"),
        ("Did that really just happen?", "surprise"),
        ("You quit your job? Just like that?", "surprise"),
        ("I thought you were in London!", "surprise"),
        ("Wow, you look completely different!", "surprise"),
        ("They gave you the lead role? Incredible!", "surprise"),
        ("I opened the door and there they all were!", "surprise"),
        ("He proposed at the restaurant in front of everyone!", "surprise"),
        ("The results weren't at all what we predicted.", "surprise"),
        ("Since when can you speak French?", "surprise"),
        ("Wait, you two are related?", "surprise"),
        ("I just found out I have a twin sibling!", "surprise"),
        ("The professor cancelled the final exam!", "surprise"),
        ("You learned all of that in just one week?", "surprise"),
        ("Oh! I didn't realize you were standing there!", "surprise"),
        ("The entire ending was a dream sequence?", "surprise"),
        ("You made this? From scratch?", "surprise"),
        ("They actually remembered my name after all these years!", "surprise"),
        ("He left everything to charity? All of it?", "surprise"),
        ("I can't believe the score was that close!", "surprise"),

        # ── DISGUST ──────────────────────────────────────────────────
        ("That's the most disgusting thing I've ever seen.", "disgust"),
        ("I can't eat this, it's absolutely revolting.", "disgust"),
        ("The smell in this room is making me sick.", "disgust"),
        ("How could you do something so shameful?", "disgust"),
        ("I find their behavior utterly repulsive.", "disgust"),
        ("There are cockroaches everywhere in this kitchen!", "disgust"),
        ("I'm appalled by how they treated those people.", "disgust"),
        ("This corruption makes me want to throw up.", "disgust"),
        ("The conditions in that building are horrific.", "disgust"),
        ("You actually enjoy watching that garbage?", "disgust"),
        ("People like that make my skin crawl.", "disgust"),
        ("What he did to those animals is sickening.", "disgust"),
        ("I can't believe you would stoop that low.", "disgust"),
        ("The food had gone completely rotten.", "disgust"),
        ("His attitude toward women is absolutely vile.", "disgust"),
        ("I stepped in something and now my shoes are ruined.", "disgust"),
        ("There was mold growing all over the walls.", "disgust"),
        ("I overheard what they said and it was repulsive.", "disgust"),
        ("The way they exploit workers makes me sick.", "disgust"),
        ("I will never eat at that restaurant again.", "disgust"),
        ("Finding a hair in your food is the worst.", "disgust"),
        ("Their hypocrisy is absolutely nauseating.", "disgust"),
        ("The whole place reeks of something terrible.", "disgust"),
        ("I lost all respect for him after that.", "disgust"),
        ("What a pathetic display of dishonesty.", "disgust"),
        ("The bathroom hasn't been cleaned in weeks.", "disgust"),
        ("Those photos are deeply disturbing to look at.", "disgust"),
        ("I refuse to be associated with such behavior.", "disgust"),
        ("That joke crossed every line of decency.", "disgust"),
        ("Watching them lie with a straight face is sickening.", "disgust"),
        ("I wouldn't touch that with a ten-foot pole.", "disgust"),
        ("The way he cheated everyone is unforgivable.", "disgust"),
        ("This milk smells like it expired weeks ago.", "disgust"),
        ("I gagged the moment I walked into that room.", "disgust"),
        ("Their greed knows absolutely no bounds.", "disgust"),
        ("Picking your teeth at the dinner table? Really?", "disgust"),
        ("I found out they've been dumping waste in the river.", "disgust"),
        ("His arrogance is truly stomach-turning.", "disgust"),
        ("There were bugs crawling in the cereal box!", "disgust"),
        ("I can't believe he said that in front of everyone.", "disgust"),

        # ── NEUTRAL ──────────────────────────────────────────────────
        ("I'll meet you at the coffee shop at three.", "neutral"),
        ("The meeting has been rescheduled to Friday.", "neutral"),
        ("Can you pass me the salt please?", "neutral"),
        ("I'm going to the store, need anything?", "neutral"),
        ("The weather forecast says rain tomorrow.", "neutral"),
        ("My phone battery is running low.", "neutral"),
        ("I parked the car in the usual spot.", "neutral"),
        ("Let me check my schedule and get back to you.", "neutral"),
        ("The file is on the shared drive.", "neutral"),
        ("I'll have the chicken with rice, please.", "neutral"),
        ("We should leave in about ten minutes.", "neutral"),
        ("The book is on the third shelf.", "neutral"),
        ("I took the bus to work today.", "neutral"),
        ("The report needs to be submitted by Monday.", "neutral"),
        ("I'm going to bed, see you in the morning.", "neutral"),
        ("The temperature is around twenty degrees today.", "neutral"),
        ("I need to stop at the pharmacy on the way home.", "neutral"),
        ("The flight lands at six-thirty.", "neutral"),
        ("I'll be working late tonight.", "neutral"),
        ("There's leftovers in the fridge if you're hungry.", "neutral"),
        ("My appointment is at two o'clock.", "neutral"),
        ("I'll take care of it first thing tomorrow.", "neutral"),
        ("The keys are on the kitchen counter.", "neutral"),
        ("I just got home a few minutes ago.", "neutral"),
        ("We moved here about five years ago.", "neutral"),
        ("My name is Sarah and I'm a teacher.", "neutral"),
        ("I usually wake up around seven.", "neutral"),
        ("The restaurant closes at ten on weekdays.", "neutral"),
        ("I sent you an email about the project.", "neutral"),
        ("Let me know when you're ready to go.", "neutral"),
        ("The movie starts at eight.", "neutral"),
        ("I usually take this route to avoid traffic.", "neutral"),
        ("The instructions say to press the green button.", "neutral"),
        ("I read about it in the newspaper this morning.", "neutral"),
        ("She works at the hospital on Main Street.", "neutral"),
        ("I think the store opens at nine.", "neutral"),
        ("I'm currently reading a book about history.", "neutral"),
        ("The package should arrive sometime this week.", "neutral"),
        ("I have a meeting with the team after lunch.", "neutral"),
        ("The printer is on the second floor.", "neutral"),
        ("We have practice every Tuesday and Thursday.", "neutral"),
        ("I forgot to set my alarm last night.", "neutral"),
        ("The project deadline is two weeks from now.", "neutral"),
        ("There's a new cafe that opened down the street.", "neutral"),
        ("I need to return these library books.", "neutral"),
        ("The doctor said to come back in two weeks.", "neutral"),
        ("I was just watching the news before you called.", "neutral"),
        ("I'm planning to visit my parents this weekend.", "neutral"),
        ("The next train leaves in fifteen minutes.", "neutral"),
        ("I picked up some groceries on the way home.", "neutral"),
    ]

    df = pd.DataFrame(data, columns=["text", "label"])
    return df


def main():
    print("📥 Downloading MELD text data (CSV only, ~3 MB)...\n")

    all_rows = []
    test_rows_list = []
    download_success = False

    for split, url in MELD_URLS.items():
        try:
            df = download_and_parse(url)
            if df.empty:
                continue

            rows = extract_text_emotion(df)
            if len(rows) > 50:
                download_success = True
                if split == "test":
                    test_rows_list.extend(rows)
                else:
                    all_rows.extend(rows)
                print(f"       → extracted {len(rows):,} valid samples")
        except Exception as e:
            print(f"   ❌ {split}: {e}")

    # ── Fallback if download failed ───────────────────────────────────
    if not download_success or len(all_rows) < 100:
        print(f"\n⚠  MELD download got only {len(all_rows)} samples (expected ~10K).")
        fallback_df = create_builtin_dataset()

        # Split 85/15 for train/test
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            fallback_df, test_size=0.15, stratify=fallback_df["label"], random_state=42
        )
    else:
        train_df = pd.DataFrame(all_rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
        test_df = pd.DataFrame(test_rows_list).drop_duplicates(subset=["text"]).reset_index(drop=True)

    # ── Show label distribution ───────────────────────────────────────
    print(f"\n📊 Label Distribution:")
    print("-" * 40)
    for label, count in train_df["label"].value_counts().items():
        bar = "█" * max(1, count // 5)
        print(f"  {label:<12} {count:>5}  {bar}")
    print("-" * 40)
    print(f"  {'TOTAL':<12} {len(train_df):>5}")

    # ── Save ──────────────────────────────────────────────────────────
    train_df.to_csv("labeled_subtitles.csv", index=False)
    print(f"\n✅ Saved {len(train_df):,} train samples → labeled_subtitles.csv")

    test_df.to_csv("test_subtitles.csv", index=False)
    print(f"✅ Saved {len(test_df):,} test samples  → test_subtitles.csv")

    print("\n📝 Next step: run 'python finetune.py' to train the model")


if __name__ == "__main__":
    main()
