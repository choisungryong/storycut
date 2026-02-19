/* ═══════════════════════════════════════════════════════════
   Klippa Landing Page Scripts
   i18n, scroll animations, counters, FAQ, gallery, mobile menu
   ═══════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  // ─── i18n Translations ───
  const translations = {
    en: {
      'nav.features': 'Features',
      'nav.gallery': 'Gallery',
      'nav.pricing': 'Pricing',
      'nav.cta': 'Start Free',

      'hero.title1': 'Your story,',
      'hero.title2': 'made into film.',
      'hero.subtitle': 'AI analyzes your text, paints the scenes, adds the voice, and delivers the video.',
      'hero.cta': 'Start Free',
      'hero.how': 'See How It Works',
      'hero.badge1': 'Videos Created',
      'hero.badge2': 'Avg Creation',
      'hero.badge3': 'Art Styles',
      'hero.badge4': 'Music Videos',
      'hero.scroll': 'Scroll to explore',

      'dual.label': 'Two Ways to Create',
      'dual.title': 'Choose your creation mode',
      'dual.subtitle': 'From text to video or music to visual story - AI handles the entire production pipeline.',
      'dual.story_tag': 'AI Story Video',
      'dual.story_title': 'AI Story Video',
      'dual.story_desc': 'Enter a topic and AI writes the story, generates scenes, and produces a complete video with narration.',
      'dual.story_f1': 'Character consistency',
      'dual.story_f2': 'B-roll auto placement',
      'dual.story_f3': '7 art styles',
      'dual.story_f4': 'Subtitles + TTS narration',
      'dual.story_cta': 'Create Story Video \u2192',
      'dual.mv_tag': 'Music Video',
      'dual.mv_title': 'Music Video',
      'dual.mv_desc': 'Upload your music and AI analyzes lyrics, BPM, and mood to create a synchronized music video.',
      'dual.mv_f1': 'AI lyrics sync',
      'dual.mv_f2': 'BPM-based scene splits',
      'dual.mv_f3': 'Visual mood matching',
      'dual.mv_f4': 'Subtitle overlay',
      'dual.mv_cta': 'Create Music Video \u2192',

      'pipe.label': 'How It Works',
      'pipe.title': 'From idea to video in 4 steps',
      'pipe.s1_title': 'Enter Your Topic',
      'pipe.s1_desc': 'Type a simple topic or detailed script. AI understands context, mood, and narrative intent.',
      'pipe.s1_demo': '"A cat walking through neon-lit Tokyo streets at midnight"',
      'pipe.s2_title': 'AI Writes the Story',
      'pipe.s2_desc': 'Gemini AI generates a multi-scene screenplay with character descriptions, camera directions, and emotional arcs.',
      'pipe.s3_title': 'Generate Scenes',
      'pipe.s3_desc': 'Each scene becomes a cinematic image with consistent characters, then transforms into dynamic video with Ken Burns and I2V effects.',
      'pipe.s4_title': 'Complete Video',
      'pipe.s4_desc': 'TTS narration, background music, subtitles, cinematic transitions, and film grain are composed into your final video.',

      'feat.label': 'Features',
      'feat.title': "What other AI video tools don't have",
      'feat.char_title': 'Character Consistency',
      'feat.char_desc': 'Master Anchor + 7-step LOCK system ensures the same character appears identically across every scene. No more face-changing AI videos.',
      'feat.char_label': 'Same character across all scenes',
      'feat.style_title': '7 Art Styles',
      'feat.style_desc': 'Cinematic, Anime, Webtoon, Realistic, Illustration, Game Anime, Abstract - each with unique visual identity.',
      'feat.broll_title': 'Auto B-roll',
      'feat.broll_desc': 'Pexels stock footage automatically placed in non-character scenes for professional quality at zero cost.',
      'feat.lyrics_title': 'AI Lyrics Sync',
      'feat.lyrics_desc': 'Upload music and AI analyzes BPM, mood, and lyrics to automatically sync visual scenes with musical structure. Gemini-powered alignment for near-perfect timing.',
      'feat.editor_title': 'Live Scene Editor',
      'feat.editor_desc': 'Review each scene, edit prompts, and regenerate individual images before final render.',
      'feat.i2v_title': 'I2V Conversion',
      'feat.i2v_desc': 'Transform still images into real video using Veo 3.1 image-to-video technology.',

      'gallery.label': 'Style Gallery',
      'gallery.title': '7 worlds, your choice',
      'gallery.subtitle': 'Each style creates a unique visual identity for your video.',
      'gallery.desc_cinematic': 'Cinematic style delivers movie-grade lighting, deep composition, and premium visual quality reminiscent of Hollywood productions.',

      'stats.created': 'Videos Created',
      'stats.avg': 'Avg Creation Time',
      'stats.success': 'Success Rate',

      'price.title': 'Start at the right price',
      'price.popular': 'Popular',
      'price.start': 'Get Started',
      'price.compare': 'View full plan comparison \u2192',
      'price.f_price': '\u20a90',
      'price.f_clips': '30 clips',
      'price.f_styles': 'Basic styles',
      'price.f_watermark': 'Watermark included',
      'price.f_720p': '720p output',
      'price.f_gemini': 'Gemini 2.5 only',
      'price.per_mo': '/mo',
      'price.l_price': '\u20a911,900',
      'price.l_clips': '150 clips',
      'price.l_nowm': 'No watermark',
      'price.l_1080p': '1080p output',
      'price.l_i2v': 'I2V included',
      'price.l_gemini': 'Gemini 3.0 x3 free/mo',
      'price.p_price': '\u20a929,900',
      'price.p_clips': '500 clips',
      'price.p_nowm': 'No watermark',
      'price.p_1080p': '1080p output',
      'price.p_priority': 'Priority processing',
      'price.p_gemini': 'Gemini 3.0 x10 free/mo',
      'price.pm_price': '\u20a999,000',
      'price.pm_clips': '2,000 clips',
      'price.pm_nowm': 'No watermark',
      'price.pm_1080p': '1080p output',
      'price.pm_priority': 'Priority processing',
      'price.pm_gemini': 'Gemini 3.0 unlimited',

      'faq.title': 'Frequently asked questions',
      'faq.q1': 'What is Klippa?',
      'faq.a1': 'Klippa is an AI-powered video creation studio that transforms text scripts or music into fully produced videos. It handles everything from story writing and image generation to narration, subtitles, and final video composition.',
      'faq.q2': 'How long does video generation take?',
      'faq.a2': 'Typically 30 seconds to 2 minutes depending on video length and number of scenes. Music videos may take slightly longer due to audio analysis and lyrics synchronization.',
      'faq.q3': 'What AI models are used?',
      'faq.a3': 'Klippa uses Google Gemini for story generation and image creation, Veo 3.1 for image-to-video conversion, ElevenLabs for TTS narration, and proprietary algorithms for scene composition and transitions.',
      'faq.q4': 'Who owns the generated videos?',
      'faq.a4': 'You retain full rights to all videos generated through Klippa. You can use them for personal, commercial, and social media purposes without restrictions.',
      'faq.q5': 'What music formats are supported for MV mode?',
      'faq.a5': 'Klippa supports MP3, WAV, M4A, and FLAC audio formats. The AI analyzes BPM, mood, energy levels, and lyrics (if present) to create perfectly synchronized visual scenes.',
      'faq.q6': 'Can I try it for free?',
      'faq.a6': 'Yes! Every new account receives 30 free clips. You can create multiple videos to experience the full pipeline before deciding on a paid plan.',
      'faq.q7': 'What is the refund policy?',
      'faq.a7': "We offer a full refund within 7 days of purchase if you're not satisfied. Unused clips can be carried over to the next billing cycle.",

      'cta.title': 'Turn your story into video, now.',
      'cta.sub': 'No account required to start.',
      'cta.btn': 'Start Free',

      'footer.desc': 'AI-Powered Video Creation Studio. Turn your stories into cinematic videos.',
      'footer.product': 'Product',
      'footer.resources': 'Resources',
      'footer.legal': 'Legal',
      'footer.terms': 'Terms of Service',
      'footer.privacy': 'Privacy Policy',
    },
    ko: {
      'nav.features': '\uAE30\uB2A5',
      'nav.gallery': '\uAC24\uB7EC\uB9AC',
      'nav.pricing': '\uC694\uAE08\uC81C',
      'nav.cta': '\uBB34\uB8CC\uB85C \uC2DC\uC791',

      'hero.title1': '\uB2F9\uC2E0\uC758 \uC774\uC57C\uAE30\uB97C',
      'hero.title2': '\uC601\uD654\uB85C \uB9CC\uB4ED\uB2C8\uB2E4.',
      'hero.subtitle': 'AI\uAC00 \uD14D\uC2A4\uD2B8\uB97C \uBD84\uC11D\uD558\uACE0, \uC7A5\uBA74\uC744 \uADF8\uB9AC\uACE0, \uBAA9\uC18C\uB9AC\uB97C \uC785\uD788\uACE0, \uC601\uC0C1\uC73C\uB85C \uC644\uC131\uD569\uB2C8\uB2E4.',
      'hero.cta': '\uBB34\uB8CC\uB85C \uC2DC\uC791',
      'hero.how': '\uC791\uB3D9 \uBC29\uC2DD \uBCF4\uAE30',
      'hero.badge1': '\uC0DD\uC131\uB41C \uC601\uC0C1',
      'hero.badge2': '\uD3C9\uADE0 \uC0DD\uC131 \uC2DC\uAC04',
      'hero.badge3': '\uC544\uD2B8 \uC2A4\uD0C0\uC77C',
      'hero.badge4': '\uBBA4\uC9C1\uBE44\uB514\uC624',
      'hero.scroll': '\uC2A4\uD06C\uB864\uD558\uC5EC \uB354 \uC54C\uC544\uBCF4\uAE30',

      'dual.label': '\uB450 \uAC00\uC9C0 \uC81C\uC791 \uBC29\uC2DD',
      'dual.title': '\uC6D0\uD558\uB294 \uC81C\uC791 \uBAA8\uB4DC\uB97C \uC120\uD0DD\uD558\uC138\uC694',
      'dual.subtitle': '\uD14D\uC2A4\uD2B8\uC5D0\uC11C \uC601\uC0C1\uC73C\uB85C, \uC74C\uC545\uC5D0\uC11C \uBE44\uC8FC\uC5BC \uC2A4\uD1A0\uB9AC\uB85C - AI\uAC00 \uC804\uCCB4 \uD504\uB85C\uB355\uC158\uC744 \uCC98\uB9AC\uD569\uB2C8\uB2E4.',
      'dual.story_tag': 'AI \uC2A4\uD1A0\uB9AC \uC601\uC0C1',
      'dual.story_title': 'AI \uC2A4\uD1A0\uB9AC \uC601\uC0C1',
      'dual.story_desc': '\uC8FC\uC81C\uB97C \uC785\uB825\uD558\uBA74 AI\uAC00 \uC2A4\uD1A0\uB9AC\uB97C \uC4F0\uACE0, \uC7A5\uBA74\uC744 \uC0DD\uC131\uD558\uACE0, \uB0B4\uB808\uC774\uC158\uACFC \uD568\uAED8 \uC644\uC131\uB41C \uC601\uC0C1\uC744 \uC81C\uC791\uD569\uB2C8\uB2E4.',
      'dual.story_f1': '\uCE90\uB9AD\uD130 \uC77C\uAD00\uC131',
      'dual.story_f2': 'B-roll \uC790\uB3D9 \uBC30\uCE58',
      'dual.story_f3': '7\uAC00\uC9C0 \uC544\uD2B8 \uC2A4\uD0C0\uC77C',
      'dual.story_f4': '\uC790\uB9C9 + TTS \uB0B4\uB808\uC774\uC158',
      'dual.story_cta': '\uC2A4\uD1A0\uB9AC \uC601\uC0C1 \uB9CC\uB4E4\uAE30 \u2192',
      'dual.mv_tag': '\uBBA4\uC9C1\uBE44\uB514\uC624',
      'dual.mv_title': '\uBBA4\uC9C1\uBE44\uB514\uC624',
      'dual.mv_desc': '\uC74C\uC545\uC744 \uC5C5\uB85C\uB4DC\uD558\uBA74 AI\uAC00 \uAC00\uC0AC, BPM, \uBB34\uB4DC\uB97C \uBD84\uC11D\uD558\uC5EC \uC2F1\uD06C\uB41C \uBBA4\uC9C1\uBE44\uB514\uC624\uB97C \uC81C\uC791\uD569\uB2C8\uB2E4.',
      'dual.mv_f1': 'AI \uAC00\uC0AC \uC2F1\uD06C',
      'dual.mv_f2': 'BPM \uAE30\uBC18 \uC528 \uBD84\uD560',
      'dual.mv_f3': '\uBE44\uC8FC\uC5BC \uBB34\uB4DC \uB9E4\uCE6D',
      'dual.mv_f4': '\uC790\uB9C9 \uC624\uBC84\uB808\uC774',
      'dual.mv_cta': '\uBBA4\uC9C1\uBE44\uB514\uC624 \uB9CC\uB4E4\uAE30 \u2192',

      'pipe.label': '\uC791\uB3D9 \uBC29\uC2DD',
      'pipe.title': '\uC544\uC774\uB514\uC5B4\uC5D0\uC11C \uC601\uC0C1\uAE4C\uC9C0 4\uB2E8\uACC4',
      'pipe.s1_title': '\uC8FC\uC81C \uC785\uB825',
      'pipe.s1_desc': '\uAC04\uB2E8\uD55C \uC8FC\uC81C\uB098 \uC0C1\uC138\uD55C \uC2A4\uD06C\uB9BD\uD2B8\uB97C \uC785\uB825\uD558\uC138\uC694. AI\uAC00 \uB9E5\uB77D, \uBD84\uC704\uAE30, \uC11C\uC0AC \uC758\uB3C4\uB97C \uC774\uD574\uD569\uB2C8\uB2E4.',
      'pipe.s1_demo': '"\uD55C\uBC24\uC911 \uB124\uC628 \uBE5B \uB3C4\uCFC4 \uAC70\uB9AC\uB97C \uAC77\uB294 \uACE0\uC591\uC774"',
      'pipe.s2_title': 'AI \uC2A4\uD1A0\uB9AC \uC0DD\uC131',
      'pipe.s2_desc': 'Gemini AI\uAC00 \uCE90\uB9AD\uD130 \uC124\uBA85, \uCE74\uBA54\uB77C \uC9C0\uC2DC, \uAC10\uC815 \uC544\uD06C\uB97C \uD3EC\uD568\uD55C \uBA40\uD2F0\uC528 \uC2DC\uB098\uB9AC\uC624\uB97C \uC0DD\uC131\uD569\uB2C8\uB2E4.',
      'pipe.s3_title': '\uC7A5\uBA74 \uC0DD\uC131',
      'pipe.s3_desc': '\uAC01 \uC7A5\uBA74\uC774 \uC77C\uAD00\uB41C \uCE90\uB9AD\uD130\uC758 \uC2DC\uB124\uB9C8\uD1B1 \uC774\uBBF8\uC9C0\uB85C \uBCC0\uD658\uB418\uACE0, Ken Burns \uD6A8\uACFC\uC640 I2V\uB85C \uC601\uC0C1\uC73C\uB85C \uC804\uD658\uB429\uB2C8\uB2E4.',
      'pipe.s4_title': '\uC601\uC0C1 \uC644\uC131',
      'pipe.s4_desc': 'TTS \uB0B4\uB808\uC774\uC158, \uBC30\uACBD\uC74C\uC545, \uC790\uB9C9, \uC2DC\uB124\uB9C8\uD1B1 \uC804\uD658, \uD544\uB984 \uADF8\uB808\uC778\uC774 \uD569\uCCD0\uC838 \uCD5C\uC885 \uC601\uC0C1\uC774 \uC644\uC131\uB429\uB2C8\uB2E4.',

      'feat.label': '\uAE30\uB2A5',
      'feat.title': '\uB2E4\uB978 AI \uC601\uC0C1 \uB3C4\uAD6C\uC5D0\uB294 \uC5C6\uB294 \uAC83\uB4E4',
      'feat.char_title': '\uCE90\uB9AD\uD130 \uC77C\uAD00\uC131',
      'feat.char_desc': 'Master Anchor + 7\uB2E8\uACC4 LOCK \uC2DC\uC2A4\uD15C\uC73C\uB85C \uBAA8\uB4E0 \uC7A5\uBA74\uC5D0\uC11C \uB3D9\uC77C\uD55C \uCE90\uB9AD\uD130\uAC00 \uB4F1\uC7A5\uD569\uB2C8\uB2E4. \uB354 \uC774\uC0C1 \uC5BC\uAD74\uC774 \uBC14\uB00C\uB294 AI \uC601\uC0C1\uC740 \uC5C6\uC2B5\uB2C8\uB2E4.',
      'feat.char_label': '\uBAA8\uB4E0 \uC7A5\uBA74\uC5D0\uC11C \uB3D9\uC77C\uD55C \uCE90\uB9AD\uD130',
      'feat.style_title': '7\uAC00\uC9C0 \uC544\uD2B8 \uC2A4\uD0C0\uC77C',
      'feat.style_desc': 'Cinematic, Anime, Webtoon, Realistic, Illustration, Game Anime, Abstract - \uAC01\uAC01 \uACE0\uC720\uD55C \uBE44\uC8FC\uC5BC \uC544\uC774\uB374\uD2F0\uD2F0.',
      'feat.broll_title': '\uC790\uB3D9 B-roll',
      'feat.broll_desc': 'Pexels \uC2A4\uD1A1 \uC601\uC0C1\uC774 \uCE90\uB9AD\uD130 \uC5C6\uB294 \uC7A5\uBA74\uC5D0 \uC790\uB3D9 \uBC30\uCE58\uB418\uC5B4 \uBE44\uC6A9 \uC5C6\uC774 \uD504\uB85C\uAE09 \uD488\uC9C8\uC744 \uC81C\uACF5\uD569\uB2C8\uB2E4.',
      'feat.lyrics_title': 'AI \uAC00\uC0AC \uC2F1\uD06C',
      'feat.lyrics_desc': '\uC74C\uC545\uC744 \uC5C5\uB85C\uB4DC\uD558\uBA74 AI\uAC00 BPM, \uBB34\uB4DC, \uAC00\uC0AC\uB97C \uBD84\uC11D\uD558\uC5EC \uC74C\uC545 \uAD6C\uC870\uC5D0 \uB9DE\uCDB0 \uC2DC\uAC01\uC801 \uC7A5\uBA74\uC744 \uC790\uB3D9 \uB3D9\uAE30\uD654\uD569\uB2C8\uB2E4.',
      'feat.editor_title': '\uC2E4\uC2DC\uAC04 \uC528 \uD3B8\uC9D1\uAE30',
      'feat.editor_desc': '\uAC01 \uC7A5\uBA74\uC744 \uB9AC\uBDF0\uD558\uACE0, \uD504\uB86C\uD504\uD2B8\uB97C \uD3B8\uC9D1\uD558\uACE0, \uCD5C\uC885 \uB80C\uB354 \uC804\uC5D0 \uAC1C\uBCC4 \uC774\uBBF8\uC9C0\uB97C \uC7AC\uC0DD\uC131\uD560 \uC218 \uC788\uC2B5\uB2C8\uB2E4.',
      'feat.i2v_title': 'I2V \uBCC0\uD658',
      'feat.i2v_desc': 'Veo 3.1 \uC774\uBBF8\uC9C0-\uD22C-\uBE44\uB514\uC624 \uAE30\uC220\uB85C \uC815\uC9C0 \uC774\uBBF8\uC9C0\uB97C \uC2E4\uC81C \uC601\uC0C1\uC73C\uB85C \uBCC0\uD658\uD569\uB2C8\uB2E4.',

      'gallery.label': '\uC2A4\uD0C0\uC77C \uAC24\uB7EC\uB9AC',
      'gallery.title': '7\uAC00\uC9C0 \uC138\uACC4\uAD00, \uB2F9\uC2E0\uC758 \uC120\uD0DD',
      'gallery.subtitle': '\uAC01 \uC2A4\uD0C0\uC77C\uC774 \uC601\uC0C1\uC5D0 \uACE0\uC720\uD55C \uBE44\uC8FC\uC5BC \uC544\uC774\uB374\uD2F0\uD2F0\uB97C \uBD80\uC5EC\uD569\uB2C8\uB2E4.',
      'gallery.desc_cinematic': 'Cinematic \uC2A4\uD0C0\uC77C\uC740 \uC601\uD654\uAE09 \uC870\uBA85, \uAE4A\uC740 \uAD6C\uB3C4, \uD560\uB9AC\uC6B0\uB4DC \uD504\uB85C\uB355\uC158\uC744 \uC5F0\uC0C1\uC2DC\uD0A4\uB294 \uD504\uB9AC\uBBF8\uC5C4 \uBE44\uC8FC\uC5BC\uC744 \uC81C\uACF5\uD569\uB2C8\uB2E4.',

      'stats.created': '\uC0DD\uC131\uB41C \uC601\uC0C1',
      'stats.avg': '\uD3C9\uADE0 \uC0DD\uC131 \uC2DC\uAC04',
      'stats.success': '\uC131\uACF5\uB960',

      'price.title': '\uD569\uB9AC\uC801\uC778 \uAC00\uACA9\uC73C\uB85C \uC2DC\uC791\uD558\uC138\uC694',
      'price.popular': '\uC778\uAE30',
      'price.start': '\uC2DC\uC791\uD558\uAE30',
      'price.compare': '\uC804\uCCB4 \uC694\uAE08\uC81C \uBE44\uAD50 \uBCF4\uAE30 \u2192',
      'price.f_price': '\u20a90',
      'price.f_clips': '30 \uD074\uB9BD',
      'price.f_styles': '\uAE30\uBCF8 \uC2A4\uD0C0\uC77C',
      'price.f_watermark': '\uC6CC\uD130\uB9C8\uD06C \uD3EC\uD568',
      'price.f_720p': '720p \uCD9C\uB825',
      'price.f_gemini': 'Gemini 2.5\uB9CC \uC0AC\uC6A9',
      'price.per_mo': '/\uC6D4',
      'price.l_price': '\u20a911,900',
      'price.l_clips': '150 \uD074\uB9BD',
      'price.l_nowm': '\uC6CC\uD130\uB9C8\uD06C \uC5C6\uC74C',
      'price.l_1080p': '1080p \uCD9C\uB825',
      'price.l_i2v': 'I2V \uD3EC\uD568',
      'price.l_gemini': 'Gemini 3.0 \uC6D4 3\uD68C \uBB34\uB8CC',
      'price.p_price': '\u20a929,900',
      'price.p_clips': '500 \uD074\uB9BD',
      'price.p_nowm': '\uC6CC\uD130\uB9C8\uD06C \uC5C6\uC74C',
      'price.p_1080p': '1080p \uCD9C\uB825',
      'price.p_priority': '\uC6B0\uC120 \uCC98\uB9AC',
      'price.p_gemini': 'Gemini 3.0 \uC6D4 10\uD68C \uBB34\uB8CC',
      'price.pm_price': '\u20a999,000',
      'price.pm_clips': '2,000 \uD074\uB9BD',
      'price.pm_nowm': '\uC6CC\uD130\uB9C8\uD06C \uC5C6\uC74C',
      'price.pm_1080p': '1080p \uCD9C\uB825',
      'price.pm_priority': '\uC6B0\uC120 \uCC98\uB9AC',
      'price.pm_gemini': 'Gemini 3.0 \uBB34\uC81C\uD55C',

      'faq.title': '\uC790\uC8FC \uBB3B\uB294 \uC9C8\uBB38',
      'faq.q1': 'Klippa\uB294 \uC5B4\uB5A4 \uC11C\uBE44\uC2A4\uC778\uAC00\uC694?',
      'faq.a1': 'Klippa\uB294 \uD14D\uC2A4\uD2B8 \uC2A4\uD06C\uB9BD\uD2B8\uB098 \uC74C\uC545\uC744 \uC644\uC131\uB41C \uC601\uC0C1\uC73C\uB85C \uBCC0\uD658\uD558\uB294 AI \uAE30\uBC18 \uC601\uC0C1 \uC81C\uC791 \uC2A4\uD29C\uB514\uC624\uC785\uB2C8\uB2E4. \uC2A4\uD1A0\uB9AC \uC791\uC131, \uC774\uBBF8\uC9C0 \uC0DD\uC131, \uB0B4\uB808\uC774\uC158, \uC790\uB9C9, \uCD5C\uC885 \uC601\uC0C1 \uD569\uC131\uAE4C\uC9C0 \uBAA8\uB4E0 \uACFC\uC815\uC744 \uCC98\uB9AC\uD569\uB2C8\uB2E4.',
      'faq.q2': '\uC601\uC0C1 \uC0DD\uC131\uC5D0 \uC5BC\uB9C8\uB098 \uAC78\uB9AC\uB098\uC694?',
      'faq.a2': '\uC77C\uBC18\uC801\uC73C\uB85C 30\uCD08~2\uBD84 \uC815\uB3C4 \uC18C\uC694\uB429\uB2C8\uB2E4. \uC601\uC0C1 \uAE38\uC774\uC640 \uC528 \uC218\uC5D0 \uB530\uB77C \uB2EC\uB77C\uC9D1\uB2C8\uB2E4. \uBBA4\uC9C1\uBE44\uB514\uC624\uB294 \uC74C\uC545 \uBD84\uC11D\uACFC \uAC00\uC0AC \uC2F1\uD06C\uB85C \uC778\uD574 \uC870\uAE08 \uB354 \uAC78\uB9B4 \uC218 \uC788\uC2B5\uB2C8\uB2E4.',
      'faq.q3': '\uC5B4\uB5A4 AI \uBAA8\uB378\uC744 \uC0AC\uC6A9\uD558\uB098\uC694?',
      'faq.a3': 'Klippa\uB294 \uC2A4\uD1A0\uB9AC \uC0DD\uC131\uACFC \uC774\uBBF8\uC9C0 \uC81C\uC791\uC5D0 Google Gemini, \uC774\uBBF8\uC9C0-\uD22C-\uBE44\uB514\uC624 \uBCC0\uD658\uC5D0 Veo 3.1, TTS \uB0B4\uB808\uC774\uC158\uC5D0 ElevenLabs, \uC7A5\uBA74 \uD569\uC131\uACFC \uC804\uD658\uC5D0 \uB3C5\uC790 \uC54C\uACE0\uB9AC\uC998\uC744 \uC0AC\uC6A9\uD569\uB2C8\uB2E4.',
      'faq.q4': '\uC0DD\uC131\uB41C \uC601\uC0C1\uC758 \uC800\uC791\uAD8C\uC740?',
      'faq.a4': 'Klippa\uB97C \uD1B5\uD574 \uC0DD\uC131\uB41C \uBAA8\uB4E0 \uC601\uC0C1\uC758 \uAD8C\uB9AC\uB294 \uC0AC\uC6A9\uC790\uC5D0\uAC8C \uC788\uC2B5\uB2C8\uB2E4. \uAC1C\uC778, \uC0C1\uC5C5, SNS \uBAA9\uC801\uC73C\uB85C \uC81C\uD55C \uC5C6\uC774 \uC0AC\uC6A9\uD560 \uC218 \uC788\uC2B5\uB2C8\uB2E4.',
      'faq.q5': 'MV \uBAA8\uB4DC\uC5D0\uC11C \uC5B4\uB5A4 \uC74C\uC545 \uD3EC\uB9F7\uC744 \uC9C0\uC6D0\uD558\uB098\uC694?',
      'faq.a5': 'MP3, WAV, M4A, FLAC \uD3EC\uB9F7\uC744 \uC9C0\uC6D0\uD569\uB2C8\uB2E4. AI\uAC00 BPM, \uBB34\uB4DC, \uC5D0\uB108\uC9C0, \uAC00\uC0AC(\uC788\uB294 \uACBD\uC6B0)\uB97C \uBD84\uC11D\uD558\uC5EC \uC644\uBCBD\uD788 \uB3D9\uAE30\uD654\uB41C \uC2DC\uAC01\uC801 \uC7A5\uBA74\uC744 \uC0DD\uC131\uD569\uB2C8\uB2E4.',
      'faq.q6': '\uBB34\uB8CC\uB85C \uCCB4\uD5D8\uD560 \uC218 \uC788\uB098\uC694?',
      'faq.a6': '\uB124! \uBAA8\uB4E0 \uC2E0\uADDC \uACC4\uC815\uC5D0 30 \uBB34\uB8CC \uD074\uB9BD\uC774 \uC81C\uACF5\uB429\uB2C8\uB2E4. \uC720\uB8CC \uD50C\uB79C\uC744 \uACB0\uC815\uD558\uAE30 \uC804\uC5D0 \uC5EC\uB7EC \uC601\uC0C1\uC744 \uC81C\uC791\uD574\uBCFC \uC218 \uC788\uC2B5\uB2C8\uB2E4.',
      'faq.q7': '\uD658\uBD88 \uC815\uCC45\uC740?',
      'faq.a7': '\uAD6C\uB9E4 \uD6C4 7\uC77C \uC774\uB0B4 \uBD88\uB9CC\uC871 \uC2DC \uC804\uC561 \uD658\uBD88\uC744 \uC81C\uACF5\uD569\uB2C8\uB2E4. \uBBF8\uC0AC\uC6A9 \uD074\uB9BD\uC740 \uB2E4\uC74C \uACB0\uC81C \uC8FC\uAE30\uB85C \uC774\uC6D4\uB429\uB2C8\uB2E4.',

      'cta.title': '\uC9C0\uAE08, \uB2F9\uC2E0\uC758 \uC774\uC57C\uAE30\uB97C \uC601\uC0C1\uC73C\uB85C \uB9CC\uB4E4\uC5B4 \uBCF4\uC138\uC694.',
      'cta.sub': '\uACC4\uC815 \uC5C6\uC774 \uBC14\uB85C \uCCB4\uD5D8 \uAC00\uB2A5',
      'cta.btn': '\uBB34\uB8CC\uB85C \uC2DC\uC791\uD558\uAE30',

      'footer.desc': 'AI \uAE30\uBC18 \uC601\uC0C1 \uC81C\uC791 \uC2A4\uD29C\uB514\uC624. \uB2F9\uC2E0\uC758 \uC774\uC57C\uAE30\uB97C \uC2DC\uB124\uB9C8\uD1B1 \uC601\uC0C1\uC73C\uB85C.',
      'footer.product': '\uC81C\uD488',
      'footer.resources': '\uB9AC\uC18C\uC2A4',
      'footer.legal': '\uBC95\uC801 \uACE0\uC9C0',
      'footer.terms': '\uC774\uC6A9\uC57D\uAD00',
      'footer.privacy': '\uAC1C\uC778\uC815\uBCF4\uCC98\uB9AC\uBC29\uCE68',
    },
  };

  // Gallery descriptions per style per language
  const galleryDescs = {
    en: {
      cinematic: 'Cinematic style delivers movie-grade lighting, deep composition, and premium visual quality reminiscent of Hollywood productions.',
      anime: 'Anime style captures vibrant colors, expressive characters, and dynamic compositions inspired by Japanese animation.',
      webtoon: 'Webtoon style features clean line art, bold colors, and panel-like compositions perfect for digital storytelling.',
      realistic: 'Realistic style produces photorealistic imagery with natural lighting, textures, and lifelike detail.',
      illustration: 'Illustration style creates artistic, hand-drawn aesthetics with rich textures and creative compositions.',
      game_anime: 'Game Anime style blends anime aesthetics with 3D rendering for a premium game-art visual identity.',
      abstract: 'Abstract style generates expressive, non-representational visuals with bold colors and experimental compositions.',
    },
    ko: {
      cinematic: 'Cinematic \uC2A4\uD0C0\uC77C\uC740 \uC601\uD654\uAE09 \uC870\uBA85, \uAE4A\uC740 \uAD6C\uB3C4, \uD560\uB9AC\uC6B0\uB4DC \uD504\uB85C\uB355\uC158\uC744 \uC5F0\uC0C1\uC2DC\uD0A4\uB294 \uD504\uB9AC\uBBF8\uC5C4 \uBE44\uC8FC\uC5BC\uC744 \uC81C\uACF5\uD569\uB2C8\uB2E4.',
      anime: 'Anime \uC2A4\uD0C0\uC77C\uC740 \uC0DD\uB3D9\uAC10 \uC788\uB294 \uC0C9\uCC44, \uD45C\uD604\uB825 \uD48D\uBD80\uD55C \uCE90\uB9AD\uD130, \uC77C\uBCF8 \uC560\uB2C8\uBA54\uC774\uC158\uC5D0\uC11C \uC601\uAC10\uC744 \uBC1B\uC740 \uC5ED\uB3D9\uC801 \uAD6C\uB3C4\uB97C \uC81C\uACF5\uD569\uB2C8\uB2E4.',
      webtoon: 'Webtoon \uC2A4\uD0C0\uC77C\uC740 \uAE54\uB054\uD55C \uC120\uD654, \uB300\uB2F4\uD55C \uC0C9\uCC44, \uB514\uC9C0\uD138 \uC2A4\uD1A0\uB9AC\uD154\uB9C1\uC5D0 \uC644\uBCBD\uD55C \uD328\uB110 \uAD6C\uB3C4\uB97C \uC81C\uACF5\uD569\uB2C8\uB2E4.',
      realistic: 'Realistic \uC2A4\uD0C0\uC77C\uC740 \uC790\uC5F0\uC2A4\uB7EC\uC6B4 \uC870\uBA85, \uD14D\uC2A4\uCC98, \uC0DD\uC0DD\uD55C \uB514\uD14C\uC77C\uB85C \uC0AC\uC9C4\uCC98\uB7FC \uC0AC\uC2E4\uC801\uC778 \uC774\uBBF8\uC9C0\uB97C \uC0DD\uC131\uD569\uB2C8\uB2E4.',
      illustration: 'Illustration \uC2A4\uD0C0\uC77C\uC740 \uD48D\uBD80\uD55C \uD14D\uC2A4\uCC98\uC640 \uCC3D\uC758\uC801 \uAD6C\uB3C4\uB85C \uC608\uC220\uC801\uC778 \uC190\uADF8\uB9BC \uBBF8\uD559\uC744 \uB9CC\uB4ED\uB2C8\uB2E4.',
      game_anime: 'Game Anime \uC2A4\uD0C0\uC77C\uC740 \uC560\uB2C8\uBA54 \uBBF8\uD559\uACFC 3D \uB80C\uB354\uB9C1\uC744 \uACB0\uD569\uD55C \uD504\uB9AC\uBBF8\uC5C4 \uAC8C\uC784 \uC544\uD2B8 \uC2A4\uD0C0\uC77C\uC785\uB2C8\uB2E4.',
      abstract: 'Abstract \uC2A4\uD0C0\uC77C\uC740 \uB300\uB2F4\uD55C \uC0C9\uCC44\uC640 \uC2E4\uD5D8\uC801 \uAD6C\uB3C4\uB85C \uD45C\uD604\uC801\uC778 \uBE44\uAD6C\uC0C1 \uBE44\uC8FC\uC5BC\uC744 \uC0DD\uC131\uD569\uB2C8\uB2E4.',
    },
  };

  // ─── i18n Engine ───
  let currentLang = localStorage.getItem('klippa_lang') || 'ko';

  function applyLanguage(lang) {
    currentLang = lang;
    localStorage.setItem('klippa_lang', lang);
    document.documentElement.lang = lang;

    const dict = translations[lang];
    if (!dict) return;

    document.querySelectorAll('[data-i18n]').forEach((el) => {
      const key = el.getAttribute('data-i18n');
      if (dict[key] !== undefined) {
        el.textContent = dict[key];
      }
    });

    // Update gallery description for current active style
    const activeTab = document.querySelector('.gallery__tab.active');
    if (activeTab) {
      const style = activeTab.dataset.style;
      const desc = document.getElementById('galleryDesc');
      if (desc && galleryDescs[lang] && galleryDescs[lang][style]) {
        desc.textContent = galleryDescs[lang][style];
      }
    }

    // Update lang toggle buttons
    const label = lang === 'ko' ? 'EN' : 'KO';
    document.querySelectorAll('.lang-toggle__label').forEach((el) => {
      el.textContent = label;
    });
  }

  // ─── Auth-Aware CTA ───
  // 로그인 상태면 CTA 버튼을 앱(/)으로 변경
  (function updateCtaForAuth() {
    const token = localStorage.getItem('token');
    if (token) {
      document.querySelectorAll('a[href="/signup.html"]').forEach((a) => {
        a.href = '/';
      });
    }
  })();

  // Init language
  applyLanguage(currentLang);

  // Toggle buttons
  document.querySelectorAll('.lang-toggle').forEach((btn) => {
    btn.addEventListener('click', () => {
      const next = currentLang === 'ko' ? 'en' : 'ko';
      applyLanguage(next);
    });
  });

  // ─── Intersection Observer: Scroll Reveal ───
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          revealObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
  );

  document.querySelectorAll('.reveal').forEach((el) => {
    revealObserver.observe(el);
  });

  // ─── Nav Scroll Effect ───
  const nav = document.getElementById('mainNav');

  function updateNav() {
    if (window.scrollY > 60) {
      nav.classList.add('scrolled');
    } else {
      nav.classList.remove('scrolled');
    }
  }

  window.addEventListener('scroll', updateNav, { passive: true });
  updateNav();

  // ─── Mobile Menu ───
  const hamburger = document.getElementById('hamburgerBtn');
  const mobileMenu = document.getElementById('mobileMenu');

  if (hamburger && mobileMenu) {
    hamburger.addEventListener('click', () => {
      hamburger.classList.toggle('active');
      mobileMenu.classList.toggle('active');
      document.body.style.overflow = mobileMenu.classList.contains('active')
        ? 'hidden'
        : '';
    });

    mobileMenu.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => {
        hamburger.classList.remove('active');
        mobileMenu.classList.remove('active');
        document.body.style.overflow = '';
      });
    });
  }

  // ─── Count-Up Animation ───
  function animateCounter(el) {
    const target = parseInt(el.dataset.count, 10);
    if (!target) return;

    const duration = 2000;
    const start = performance.now();

    function update(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - (1 - progress) * (1 - progress);
      const current = Math.floor(eased * target);
      el.textContent = current.toLocaleString();

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        el.textContent = target.toLocaleString();
      }
    }

    requestAnimationFrame(update);
  }

  const counterObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          counterObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.5 }
  );

  document.querySelectorAll('[data-count]').forEach((el) => {
    counterObserver.observe(el);
  });

  // ─── FAQ Accordion ───
  document.querySelectorAll('.faq__question').forEach((btn) => {
    btn.addEventListener('click', () => {
      const item = btn.closest('.faq__item');
      const isActive = item.classList.contains('active');

      document.querySelectorAll('.faq__item.active').forEach((openItem) => {
        openItem.classList.remove('active');
      });

      if (!isActive) {
        item.classList.add('active');
      }
    });
  });

  // ─── Style Gallery Tabs ───
  const styleHues = {
    cinematic: 30, anime: 330, webtoon: 150, realistic: 210,
    illustration: 45, game_anime: 260, abstract: 280,
  };

  document.querySelectorAll('.gallery__tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.gallery__tab').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');

      const style = tab.dataset.style;
      const desc = document.getElementById('galleryDesc');
      const cards = document.getElementById('galleryCards');

      if (desc && galleryDescs[currentLang] && galleryDescs[currentLang][style]) {
        desc.style.opacity = '0';
        setTimeout(() => {
          desc.textContent = galleryDescs[currentLang][style];
          desc.style.opacity = '1';
        }, 150);
      }

      if (cards && styleHues[style] !== undefined) {
        const hue = styleHues[style];
        cards.querySelectorAll('.gallery__card').forEach((card, i) => {
          card.style.setProperty('--card-hue', hue + i * 5);
        });
      }
    });
  });

  // ─── Smooth Scroll for Anchor Links ───
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener('click', (e) => {
      const target = document.querySelector(anchor.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
})();
