# Rules-only baseline evaluation

Run date: 2026-10-22. The existing rules-only classifier was applied locally to the supplied recordings. No threshold, weight, or regex was changed after seeing these results. No cloud or other external service was called. The dataset files were left unchanged.

The class mapping is a **provisional assumption**. The organiser has not confirmed it.

| Folder | Reference class used only for scoring |
|---|---|
| excellent | Excellent |
| good | Good |
| average_ | Average |
| weak | Poor |
| quality issue_ | Poor |
| demo_ | Unlabelled, excluded |

Folder names, filenames, and reviewer comments were not inputs to the classifier. They were used only to pair each recording with its transcript and to score the prediction afterwards. Reports below are aggregates. They do not identify a recording, teacher, or student.

## Results

| Measure | Result |
|---|---|
| Labelled recordings evaluated | 10 |
| Unlabelled recordings excluded | 2 |
| Exact accuracy | **5/10 (50.0%)** |
| Within one class | **9/10 (90.0%)** |
| Insufficient evidence | **0/10** |

"Within one class" counts an exact match and a miss of one step on the scale Poor, Average, Good, Excellent. A missing prediction would not count. None were missing.

Confusion matrix. The row is the reference class. The column is the prediction.

| Reference | Poor | Average | Good | Excellent | No prediction |
|---|---:|---:|---:|---:|---:|
| Poor | 1 | 2 | 1 | 0 | 0 |
| Average | 1 | 0 | 1 | 0 | 0 |
| Good | 0 | 0 | 2 | 0 | 0 |
| Excellent | 0 | 0 | 0 | 2 | 0 |

Per class:

| Reference | Correct |
|---|---|
| Poor | 1/4 |
| Average | 0/2 |
| Good | 2/2 |
| Excellent | 2/2 |

On the five inexact predictions, the assessed criteria rated 0 were:

| Reason code | Misses on which it occurred |
|---|---:|
| homework_rated_0 | 4 |
| inclusion_rated_0 | 3 |
| closure_rated_0 | 1 |

A miss can contribute more than one code, so these counts overlap. Tone, camera presence, and branding do not appear here: those criteria are not assessed, and an unassessed criterion is not stored as a reason for a miss.

## What this does and does not show

The top of the scale matched on this sample: both Good recordings and both Excellent recordings were exact. That is 4 recordings. It is not a stable accuracy estimate.

The bottom and the middle did not. Only 1 of 4 Poor recordings was exact. Both Average recordings were missed, one toward Poor and one toward Good. The single error of more than one class was a Poor recording predicted as Good.

This matches the limitation already built into the rules. The scored criteria are opening, inclusion (student share of turns), closure, homework mentions, and a narrow audio check. Tone, whether the teacher is on camera, lighting, and the branded background are excluded rather than scored. Those are the signals the reviewer comments used most often to mark a class as weak or as having a quality problem. A recording can therefore look strong on the transcript rules and still be Poor by the organiser's standard. The rules are not failing to notice homework or turn-taking so much as they are not looking at the criteria that separate the lower labels.

`homework_rated_0` is the most common penalty among the misses. It means the homework phrase list did not match. It does not, by itself, mean the phrase list is wrong, and these 10 recordings are not a reason to edit it.

No recording was left unclassified. The "at least 40 points of assessed weight" rule was met for all 10, because each one had a transcript and measurable audio.

## Limitations

- Ten labelled recordings, two per Good, Average, and Excellent, and four mapped to Poor. One more recording can move a per-class result by a large amount.
- The mapping of "weak" and "quality issue" to Poor is unconfirmed.
- Each reference label comes from one short reviewer comment. There is no second rater.
- The two demo recordings were not scored and are not in the accuracy.
- The classifier does not assess tone, camera presence, lighting, or branding.
- Silence, a dark picture, and a static picture are measured and then not used as penalties, on purpose.
- Reason codes are not an error analysis of individual classes. They only say which scored checks were zero when the final class was already wrong.

## Next steps

Do not fit weights or cutoffs to these 10 recordings. The evaluation stays separate from the production rubric.

The useful next addition is the criteria this run could not score: tone, camera presence, and setup. That is the rubric judge, and it should stay off until the organiser permits Ollama Cloud. Until then, the rules-only report should keep saying those criteria were not assessed, which is what this baseline measured.
