#!/usr/bin/env python3
"""seed_product_voxi.py — Seed VOXI product cells into Tee's substrate."""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from datag_bridge import DataGBridge

bridge = DataGBridge.get()
if not bridge.ready:
    print('ERROR: bridge not ready'); sys.exit(1)

CELLS = [
    ("what is voxi", "VOXI is a Conversational Intelligence Engine that mimics human reasoning to handle communication across inbound and outbound voice calls and interactive web avatars. It engages in natural dialogue with no rigid menus and no scripts, capturing intent and context to execute tasks autonomously."),
    ("what does voxi do", "VOXI picks up every call instantly, has a natural conversation with the caller, qualifies the lead, books meetings by cross-referencing calendars, updates your CRM, and sends you a detailed call summary while the lead is still warm."),
    ("how does voxi work", "VOXI works in three stages: it engages the caller in natural dialogue so no lead is lost to hold times or voicemail, it executes actions like booking meetings and updating CRMs, and it summarises the interaction and delivers it to your inbox immediately after the call."),
    ("is voxi better than ivr", "VOXI is nothing like a traditional IVR. IVR gives callers rigid menus and frustrating prompts. VOXI has a real conversation, understands what the caller wants, and acts on it. No menu trees. No press 1 for sales."),
    ("what makes voxi different from voicemail", "Voicemail is passive. It waits for people to leave a message that may never get followed up. VOXI is active. It picks up, qualifies the lead, books the meeting, and sends you the summary before the caller has even hung up. Zero wait. Zero loss."),
    ("how do i set up voxi", "Five steps and you are live. Step 1: sign up on the platform. Step 2: define your brand personality and tone. Step 3: clone your voice or choose a professional profile. Step 4: upload your knowledge base. Step 5: forward your existing calls to your VOXI number. Done."),
    ("what are the 5 steps to set up voxi", "The five steps are: instant signup, define your brand personality, voice configuration, data ingestion, and call forwarding. Each step builds on the last and most businesses are live same day."),
    ("how long does voxi take to set up", "Most businesses are live same day. The five steps are designed to be completed without a technical team."),
    ("do i need technical skills to set up voxi", "No. The setup is designed for business owners, not developers. You define your brand personality, upload your documents, and forward your calls. VOXI handles the rest."),
    ("how do i connect voxi to my phone", "Step 5 of setup is call forwarding. You connect your existing lines to the VOXI platform. No new hardware required. Your number stays the same. Calls route through VOXI before reaching you."),
    ("can voxi handle multiple calls at once", "Yes. VOXI scales infinitely. It handles concurrent voice, chat, and avatar interactions without you needing to increase headcount. One agent or a thousand, the response time stays the same."),
    ("what is voxis knowledge brain", "VOXI Unified Knowledge Brain consolidates your SOPs, FAQs, and documentation into a live intelligence layer. The agent reasons over your actual business information in real-time, not a generic script."),
    ("can voxi book meetings", "Yes. VOXI cross-references your calendar and books meetings directly during the call. The caller gets confirmation. You get the appointment. No back-and-forth, no manual scheduling."),
    ("can voxi update my crm", "Yes. VOXI triggers workflows, updates CRM records, and syncs data with your internal systems in real-time. Every interaction is captured automatically."),
    ("what channels does voxi work on", "VOXI is multimodal. It operates across voice via SIP and PBX, web avatars for face-to-face digital interactions, and messaging including WhatsApp. One intelligence layer across all touchpoints."),
    ("does voxi work on whatsapp", "Yes. VOXI operates across voice, web avatars, and messaging channels including WhatsApp. The same intelligence layer handles all of them."),
    ("will voxi stay on brand", "Yes. You configure tone guardrails during setup. VOXI stays within those boundaries on every call. Consistent tone, consistent messaging, every time."),
    ("what happens after a voxi call", "VOXI delivers a detailed call summary to your inbox while the lead is still warm. You see who called, what they needed, what was booked, and any follow-up actions triggered."),
    ("who is voxi for", "VOXI is built for organisations that need a digital workforce to manage high-volume customer touchpoints. That includes digital receptionists, customer support, outbound sales, lead qualification, debt collection, and specialised advisory roles."),
    ("can voxi do outbound sales", "Yes. VOXI runs outbound sales agents that qualify leads, handle objections, and book meetings at scale without a human rep on every call. Your sales team focuses on closing; VOXI handles the top of funnel."),
    ("can voxi replace my receptionist", "VOXI acts as a digital receptionist that never takes a day off, never puts anyone on hold, and handles every call with the same professionalism. It qualifies, routes, books, and follows up automatically."),
    ("what industries use voxi", "VOXI is used across sales and lead generation, customer support, legal assistance, medical advisory, academic tutoring, debt collection, and scheduling. Any business with high call volume and a need for consistent intelligent responses."),
    ("how much does voxi cost", "VOXI operates on a clear commercial model. You provide your business data and policy inputs; VOXI manages platform delivery and performance. Carrier and messaging fees are agreed individually to keep billing predictable. Speak to us based on your call volume and use case."),
    ("what is the pricing model for voxi", "Partners provide the business knowledge and policy inputs. VOXI handles the platform, delivery, and performance. Usage fees for carriers and messaging are agreed individually so billing stays predictable with no surprise overages."),
    ("is voxi expensive", "Compare it to the cost of a missed call. Every lead that goes to voicemail is revenue at risk. VOXI eliminates that loss. The ROI conversation is straightforward once you know your call volume and close rate."),
    ("i already have a receptionist", "A human receptionist handles one call at a time, takes breaks, and has off days. VOXI handles unlimited concurrent calls with zero variation in quality. It works alongside your team, taking the volume so they can focus on what needs a human touch."),
    ("what if voxi says something wrong", "VOXI only works from the knowledge base you upload and the tone guardrails you configure. It cannot go off-script or make things up beyond what you have given it. Brand-safe by design."),
    ("we already use an ivr system", "IVR systems frustrate callers with rigid menus. Most people hang up before they get what they need. VOXI replaces that friction with natural dialogue. Callers get answers, you get qualified leads. The comparison is not close."),
    ("i am not sure voxi is right for us", "Tell me about your current call volume and what happens when you miss a call. That is usually where the answer is. VOXI is right for any business where a missed call means lost revenue."),
]

print(f'Seeding {len(CELLS)} VOXI product cells...')
added = 0
for description, content in CELLS:
    cid = bridge.save_cell(
        description=description,
        content=content,
        source_table='product_voxi',
        confidence=0.99,
        energy=180.0,
        internal=False,
    )
    print(f'  + {cid[:40]}  {description!r}')
    added += 1

print(f'\nDone. {added} cells added.')
print(f'Total cells now: {len(bridge.substrate.methodology_cells)}')
