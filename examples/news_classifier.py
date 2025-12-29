"""
Suppose you have some texts of news and know their categories.
You want to train a system with this pre-categorized/pre-classified
texts. So, you have better call this data your training set.
"""
from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer_multi import TrainerMulti
from naiveBayesClassifier.classifier_multi import ClassifierMulti

news_trainer = TrainerMulti(tokenizer.Tokenizer())

# You need to train the system passing each text one by one to the trainer module.
news_set = [
    {"title": "Moderation Beats Extremes",
     "text": "Not eating too much helps, but sustainable weight loss also needs balanced nutrition and consistent habits.",
     "category": "health"},
    {"title": "Stretch Before You Start",
     "text": "Warm up your joints and do light stretching before intense workouts to reduce injury risk.",
     "category": "health"},
    {"title": "Sleep and Appetite",
     "text": "Poor sleep can make you crave sugary snacks and eat more than you planned.", "category": "health"},
    {"title": "Hydration Matters",
     "text": "Sometimes thirst feels like hunger; drink water first before reaching for snacks.", "category": "health"},
    {"title": "Small Steps Count",
     "text": "A 20-minute walk every day is easier to stick with than occasional extreme workouts.",
     "category": "fitness"},
    {"title": "Strength Training Basics",
     "text": "Progressive overload—gradually increasing weight or reps—builds muscle more reliably than random routines.",
     "category": "fitness"},
    {"title": "Mindful Eating",
     "text": "Eating slowly and paying attention to fullness signals can prevent overeating.", "category": "wellness"},
    {"title": "Budget Rule of Thumb",
     "text": "Try the 50/30/20 rule: needs, wants, and savings to keep spending under control.", "category": "finance"},
    {"title": "Emergency Fund First",
     "text": "Build an emergency fund covering three to six months of expenses before taking bigger investment risks.",
     "category": "finance"},
    {"title": "Interest Compounds",
     "text": "Compounding rewards patience; small monthly contributions can grow significantly over years.",
     "category": "investing"},
    {"title": "Remote Work Routine",
     "text": "Set fixed work hours and a dedicated workspace to avoid burnout while working from home.",
     "category": "productivity"},
    {"title": "Deep Work Block",
     "text": "Turn off notifications and schedule 60–90 minutes for focused work to improve output quality.",
     "category": "productivity"},
    {"title": "Meeting Hygiene",
     "text": "If a meeting has no agenda or decision to make, consider replacing it with an async update.",
     "category": "work"},
    {"title": "Code Review Tips",
     "text": "Comment on correctness and clarity, and suggest small improvements without rewriting everything.",
     "category": "software"},
    {"title": "Naming Is Design",
     "text": "Clear variable and function names reduce bugs because the code explains itself.", "category": "software"},
    {"title": "Caching Tradeoffs",
     "text": "Caching improves speed, but you must plan invalidation or you risk serving stale data.",
     "category": "engineering"},
    {"title": "Model Overfitting",
     "text": "If training accuracy rises while validation accuracy drops, your model may be overfitting.",
     "category": "machine_learning"},
    {"title": "Data Quality Wins",
     "text": "Cleaning labels and removing duplicates often improves performance more than changing algorithms.",
     "category": "machine_learning"},
    {"title": "Photography Light",
     "text": "Soft window light can make portraits look better than harsh overhead lighting.",
     "category": "photography"},
    {"title": "Travel Packing",
     "text": "Pack versatile layers and comfortable shoes; you can always re-wear outerwear but not sore feet.",
     "category": "travel"},
    {"title": "Language Practice",
     "text": "Daily short sessions beat weekly cramming; spaced repetition helps vocabulary stick.",
     "category": "education"},
    {"title": "Study Without Burnout",
     "text": "Use the Pomodoro technique: 25 minutes focus, 5 minutes break, and longer breaks every four rounds.",
     "category": "education"},
    {"title": "Simple Dinner Plan",
     "text": "A balanced plate can be: half vegetables, a quarter protein, and a quarter whole grains.",
     "category": "food"},
    {"title": "Coffee Brewing",
     "text": "Grind size changes extraction—too fine can taste bitter, too coarse can taste sour.", "category": "food"},
    {"title": "Beginner Guitar", "text": "Practice chord transitions slowly; speed comes naturally after accuracy.",
     "category": "music"},
    {"title": "Movie Recommendation",
     "text": "If you like slow-burn mysteries, look for films with character-driven plots rather than action-heavy pacing.",
     "category": "entertainment"},
    {"title": "Team Collaboration",
     "text": "Assume good intent, ask clarifying questions early, and write decisions down to avoid confusion later.",
     "category": "communication"},
    {"title": "Customer Support Tone",
     "text": "Acknowledge the issue, summarize the user's goal, then propose the next step with clear timing.",
     "category": "customer_service"},
    {"title": "Climate Basics",
     "text": "Reducing energy waste at home—like better insulation—can cut emissions and save money.",
     "category": "environment"},
    {"title": "City Transport",
     "text": "Reliable public transit reduces traffic congestion more effectively than adding lanes indefinitely.",
     "category": "urban_planning"},
    {"title": "Election Coverage",
     "text": "Political debates often highlight values and priorities; compare candidates by policies and track records.",
     "category": "politics"},
    {"title": "Diplomacy Over Escalation",
     "text": "International conflicts can de-escalate when parties maintain communication channels and agree on ceasefire terms.",
     "category": "politics"},
    {"title": "Privacy Settings",
     "text": "Review app permissions regularly; many apps don't need constant location access to function.",
     "category": "technology"},
    {"title": "Password Hygiene",
     "text": "Use a password manager and enable two-factor authentication to reduce account takeover risk.",
     "category": "security"},
]

for news in news_set:
    news_trainer.train(news['title'] + ' ' + news['text'], news['category'])

# When you have sufficient trained data, you are almost done and can start to use
# a classifier.
news_classifier = ClassifierMulti(news_trainer.data, tokenizer.Tokenizer())

# Now you have a classifier which can give a try to classifiy text of news whose
# category is unknown, yet.
classification = news_classifier.classify(
    [  # health / fitness
        {"title": "Eat Less Sugar",
         "text": "Cutting sugary drinks and keeping balanced meals can help lose weight over time.",
         "category": "health"},
        {"title": "Daily Cardio Plan",
         "text": "A 20 minute run and light stretching improves endurance and keeps you consistent.",
         "category": "fitness"},
        {"title": "Sleep and Appetite", "text": "Poor sleep can increase hunger and make you snack more than usual.",
         "category": "health"},

        # politics / world news
        {"title": "Ceasefire Negotiations",
         "text": "Leaders discussed a temporary ceasefire and humanitarian corridors during talks.",
         "category": "politics"},
        {"title": "Sanctions Debate",
         "text": "Lawmakers argued about new sanctions and the impact on trade and energy prices.",
         "category": "politics"},

        # finance / investing
        {"title": "Budgeting Habit", "text": "Automate monthly savings and track expenses to build an emergency fund.",
         "category": "finance"},
        {"title": "Long Term Investing",
         "text": "Small contributions compounded over years can grow into a large portfolio.", "category": "investing"},

        # tech / security / software
        {"title": "App Permissions",
         "text": "Review location and microphone permissions to reduce privacy risk on your phone.",
         "category": "security"},
        {"title": "Cache Invalidation",
         "text": "Caching speeds up requests but stale cache can cause incorrect results without invalidation.",
         "category": "software"},
        {"title": "Model Overfitting",
         "text": "When validation loss rises while training loss falls, the model may be overfitting.",
         "category": "machine_learning"},

        # food / music / travel
        {"title": "Easy Pasta", "text": "Cook garlic with olive oil and add vegetables for a quick weeknight pasta.",
         "category": "food"},
        {"title": "Guitar Practice", "text": "Practice chord transitions slowly and focus on clean sound before speed.",
         "category": "music"},
        {"title": "Packing Tips", "text": "Pack light layers and comfortable shoes for walking-heavy trips.",
         "category": "travel"},

        # edge cases: very short / empty-ish
        {"title": "Workout", "text": "exercise", "category": "fitness"},
        {"title": "", "text": "not to eat too much is not enough to lose weight", "category": "health"},
        # 测试空白输入处理
        {"title": "   ", "text": "   ", "category": "unknown"}, ])

# the classification variable holds the detected categories sorted
for list in classification:
    print(list)
