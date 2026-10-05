# Recording Script — round 2 (krishiv, mary)

**30 new clips per speaker** (60 total), all reading the **same 30
sentences** below. Keep the existing 10 clips each in `data/genuine/<speaker>/`
as well; they become extra enrollment data.

## Why 30, and why the same sentences

- **Clips 1–20 are for enrollment.** The detector estimates a mean and
  spread for each of 11 features. With about 9 clips per fold those spreads
  were noisy, and 20+ makes them much steadier.
- **Clips 21–30 are held out for testing.** They are never used to enroll.
  They give a test set of genuine clips that the model has never seen,
  instead of leave-one-out.
- **Both speakers read the same text.** Prosody depends on what is said
  (how long the sentence is, where the commas fall). Using identical text
  means any difference comes from the speaker's habits, not the sentence.
  The TTS clones should be generated from **sentences 21–30** as well, so
  each genuine test clip is compared against a clone of the same sentence.
- **Many sentences have commas or lists.** In round 1 krishiv had zero
  internal pauses in 8 of 10 clips, which made the pause features nearly
  constant. These sentences give natural places to pause.

## How to record

- **Folder and naming.** Use one sentence per file, named
  `<speaker>_sNN.m4a`, for example `krishiv_s07.m4a`. Save it straight into
  `data/genuine/<speaker>/`. **Check the folder before saving**: four
  round-1 clips ended up under the wrong speaker.
- **Same setup each time.** Use the same phone or mic, the same quiet room,
  and the same distance from the mic (about 20–30 cm).
- **Read naturally**, at your normal pace. Don't over-enunciate or perform.
  If you stumble, re-record that sentence.
- **Length.** Each clip should be about 3–10 s. Leave about ½ s of silence
  before and after. Clips longer than 15 s are skipped by the pipeline.
- **Spread over two sessions if possible**, e.g. sentences 1–15 one day and
  16–30 another. Real enrollment spans days, and that variation is part of
  being genuine.

## Enrollment sentences (s01–s20)

1. I usually take the bus to college, but today I decided to walk instead.
2. Could you please send me the notes from yesterday's lecture before tonight?
3. The library closes at nine on weekdays, and at six on Saturdays.
4. We need milk, bread, eggs, and a packet of tea from the shop.
5. Honestly, I didn't expect the exam to be that difficult.
6. My phone battery died halfway through the call, so I had to call back later.
7. If it rains tomorrow, we'll move the meeting to the seminar hall.
8. She told me the train was late again, which wasn't surprising at all.
9. The project deadline is next Friday, so let's finish the report by Wednesday.
10. Have you ever noticed how quiet the campus gets during the holidays?
11. I'll be there in about twenty minutes, depending on the traffic.
12. First, open the file; then, check the numbers; finally, save a copy.
13. My grandmother makes the best tea I've ever tasted, without any doubt.
14. The weather was so pleasant this morning that I skipped breakfast and went for a run.
15. Wait, did you say the class was cancelled, or just moved to the afternoon?
16. Three hundred and forty-five students registered for the event this year.
17. I'm not sure I agree with that, but I can see why you might think so.
18. After dinner, we usually sit outside and talk for an hour or so.
19. Please remember to switch off the lights and lock the door when you leave.
20. The results surprised everyone, including the people who designed the experiment.

## Held-out test sentences (s21–s30): also the text for the TTS clones

21. I called the bank twice this morning, and nobody answered either time.
22. Can you believe how quickly this semester has gone by?
23. Let's meet near the main gate at half past four, if that works for you.
24. He forgot his umbrella, his wallet, and his keys, all on the same day.
25. Actually, I think the second option is better, although it costs a little more.
26. The new canteen menu has more variety, but the prices have gone up too.
27. Before you submit the form, double-check your name and your roll number.
28. We waited for almost an hour, and then the bus finally arrived, completely full.
29. Why does the printer always stop working right before a deadline?
30. Thank you for your help today; I really couldn't have done it without you.

## When generating clones

- Use only enrollment audio (s01–s20 plus the round-1 clips) as the voice
  sample uploaded to the cloning service. **Never upload s21–s30.**
- Generate one clone clip for each of sentences 21–30, for each speaker.
  Save them as `data/synthetic/<speaker>/<tts_system>/<speaker>_sNN.<ext>`,
  where `<tts_system>` is e.g. `elevenlabs` or `xtts`.
