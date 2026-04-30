Riverline Hiring Assignment - Voice AI Engineer
Before You Begin
A note on AI tools. You may use any AI coding assistant, including Claude, Copilot, or GPT. We assume you will. This changes what we are evaluating. We are not testing whether you can produce code. We are testing whether you understand what you built, can defend your decisions under live questioning, can modify your system under novel constraints on the spot, and made genuine engineering trade-offs that reflect judgment, not generation. Your submission will be followed by a live technical session with no AI assistance. If you cannot navigate your own codebase fluently, explain your model choices, and adapt your system live, your submission quality is irrelevant.
General Guidelines
Keep your code short and concise.
Feel free to use any tools that help you solve this assignment.
The timeline is tight. Starting immediately will help.
Make suitable assumptions wherever things are not defined.
The submission details are at the end of this document.
Reach out to us if you have any queries (jayanth@riverline.ai)
The deadline is strict. No extensions.
Challenge: Hinglish Voice Collection Agent
You are building a voice agent for post-default debt collection in India. Most Indian borrowers naturally code-switch between English and Hindi within a single conversation, sometimes within a single sentence. Our current agent breaks down on this.
Three specific failure modes we see:
False language switches. Agent flips to Hindi on a single Hindi word, background noise, or a filler like "haan".
Perceived latency spikes. When the agent switches languages, it pauses 2 to 3 seconds. The borrower says "hello, are you there?" and hangs up.
Numeric fact corruption. The settlement amount changes across a language switch. We have seen ₹50,000 become ₹35,000 after a borrower says "pachas thousand".
Your job is to build a voice agent that handles these three problems. The scenario is a debt collection call for a ₹50,000 personal loan default. The agent is negotiating a settlement. The borrower speaks Hinglish naturally and occasionally switches mid-number. You can either use Pipecat or LiveKit.
Targets
Your submission must demonstrably meet the following:
Metric
Target
How we measure
“Perceived” latency per turn (you can define what perceived latency means - it’s different from latency!)
Under 1 second
Timestamped audio logs
Language detection accuracy
Above 95%
Measured on your test corpus
Numeric fact preservation across language switches
Above 99%
Audit of recorded conversations
False switch rate on non-linguistic audio
Under 2%
Noise, filler words, disfluencies
Constraints
At least one open-source component must be in the pipeline, either STT, TTS, or LLM. You cannot ship a pure API wrapper. Banks do not allow it.
Must work end-to-end with real microphone input and real speaker output. Text simulation of voice does not count.
Must run locally. Docker Compose or equivalent. Single command to start.
No external hosting. We will run it on our machine.
Deliverables
Working Voice Agent
Local demo. Microphone in, speaker out. Handles Hinglish code-switching end to end.
Recorded Demo
One recorded conversation, minimum 90 seconds, with at least three code switches between English and Hindi. Must include one mid-number language switch (example: "I can pay pachas thousand rupees").
Measurement Report
Baseline pipeline (commercial STT and TTS, example: Deepgram plus OpenAI TTS) compared against your implementation with at least one open-source swap. Both measured on the same 20-minute test corpus. Report all four target metrics with methodology.
Architecture Writeup (1 to 2 pages)
What you built
Which STT, TTS, and language detection models you evaluated
Which you rejected and why
How you handled the mid-number switch specifically
What trade-offs you made between latency and accuracy
Decision Journal
Must include:
What you tried first and why it did not work
Model comparisons you ran (with real numbers)
Pivots you made and why
One thing you intentionally chose not to build
This is mandatory and must not be AI-generated. We will verify.
Evaluation Criteria
Weight
Criteria
30%
Working system. Real demo, real audio, real latency numbers.
25%
Measurable improvement over baseline on the four target metrics.
15%
Architecture depth. Model choices defended with empirical data.
15%
Trade-off analysis. Which levers moved which metrics.
10%
Decision journal authenticity. Real friction, real experiments.
5%
Open-source integration quality.
Rules
Language: Python
STT, TTS, LLM: Any mix, but at least one must be open-source
No starter kit. Build from scratch using Pipecat or LiveKit.
Submission Checklist
A public GitHub repo with a detailed README
Docker Compose setup that runs the full system
Recorded demo (audio file or video, minimum 90 seconds)
Measurement report with raw data
Architecture writeup
Decision journal
