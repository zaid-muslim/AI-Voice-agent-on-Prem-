# Riverside General — AI Phone Receptionist

## What this is

An AI system that answers the hospital's phone line automatically. A
caller can book, cancel, or reschedule an appointment, ask general
questions about the hospital, and speak with a natural-sounding voice —
all without a human receptionist needing to pick up.

If someone describes a real medical emergency on the call, the system
recognizes it immediately and gives them emergency guidance directly —
it does not try to "chat" its way through an emergency, and it does not
rely on the AI's judgment for that specific decision. That check happens
first, before anything else, every single call.

---

## The three things you can open in a web browser

### 1. The caller experience
This is what a patient sees and hears when they call in — a normal
phone-style conversation. You can test it yourself from a browser to
hear exactly what a real caller would experience.

### 2. The hospital admin page
This is where hospital staff manage the information the AI actually
knows and uses:
- Add, remove, or edit doctors and their specialties
- Set department hours
- View and manage bookings

Changes made here show up in what the AI tells callers — if a doctor's
hours change, the AI's answers change too, automatically.

### 3. The developer console
A separate page, meant for technical staff only, for choosing which
"brain" (AI model) and which "voice" the system uses, and for comparing
how fast different combinations respond. Most people at the hospital
will never need to open this page.

---

## How a call actually works, in plain terms

1. The caller dials in and is greeted by the AI receptionist.
2. What they say is turned into text (that's the "speech-to-text" step).
3. That text is checked **first** for anything that sounds like a real
   emergency. If it matches, the AI immediately gives emergency guidance
   — it skips its normal "thinking" step entirely for that message, on
   purpose, so there's no risk of the AI trying to be clever about a
   life-threatening situation.
4. If it's not an emergency, the AI's language model decides how to
   respond — checking real appointment availability, looking up hospital
   information, or just chatting naturally.
5. The AI's response is turned back into a natural-sounding spoken voice
   and played back to the caller.

Steps 2 through 5 typically take a little over one second combined —
noticeably fast for a fully automated system, though a little slower
than talking to a human who's already mid-sentence in their head. Some
of that time is intentional: the system deliberately waits a beat after
you stop talking, the same way a polite human listener would, so it
doesn't talk over you if you pause mid-sentence.

---

## Where the "thinking" and "talking" actually happen

The system is split across two computers behind the scenes:

- **Computer 1** does the listening, the actual decision-making
  (checking appointments, looking things up, deciding what to say), and
  runs the main phone system.
- **Computer 2** is dedicated purely to generating the spoken voice —
  kept running and "warmed up" so that turning text into speech happens
  quickly, no matter how many callers are on the line at once.

This split exists so that generating speech for many simultaneous
callers doesn't slow down the "thinking" part of the system, and
vice versa.

---

## What's proven versus what's still being tested

Not every voice or AI-model option in the developer console has been
used in a real, live phone call yet. Each option is honestly labeled:

- **Verified** — has actually been tested on a real call, with a real
  caller, and worked.
- **Candidate** — the underlying technology is real and has been
  checked carefully, but hasn't yet been proven on a real live call.

This labeling exists so that whoever picks an option from the developer
console knows exactly how much to trust it before relying on it for a
real patient.

---

## Licensing note, in plain terms

A couple of the optional voice technologies come with legal usage terms
that matter for a hospital (a real business) using them commercially.
Specifically, one optional voice option is licensed in a way that
requires legal review before using it in a paid, commercial setting —
this is clearly flagged wherever that option appears, so nobody
accidentally turns it on without checking with legal/compliance first.

---

## What to do if something isn't working

- If the call button doesn't connect at all, the most common cause is
  that one of the two computers running the system isn't turned on or
  isn't running the software — this needs a technical person to check.
- If the AI's voice sounds wrong, silent, or delayed, this is almost
  always a technical/network issue between the two computers, not a
  sign the AI itself is broken — worth reporting to whoever manages the
  technical setup rather than assuming the AI "isn't working."
- If the AI gives wrong information about a doctor, department, or
  hours, check the hospital admin page first — that information is
  editable by hospital staff and is very likely just out of date, not a
  bug in the AI itself.