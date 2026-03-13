"""
Personalized stress solutions based on predicted stress level and user inputs.
"""


def get_solutions(stress_level_label, confidence, form_data):
    """
    Returns a list of personalized solution dicts with title, description, icon.
    stress_level_label: 'Low', 'Medium', or 'High'
    confidence: float 0-1
    form_data: dict of form inputs for personalization
    """
    solutions = []

    # Always include meditation tips (helpful for all levels)
    solutions.append({
        'title': 'Meditation & Mindfulness',
        'description': 'Practice 10 minutes of deep breathing or guided meditation daily. Apps like Headspace or Calm can help.',
        'icon': 'bi-heart-pulse',
        'priority': 1
    })

    # Sleep-related (personalized)
    sleep = str(form_data.get('sleep_quality', '')).lower()
    if sleep in ('poor', '1'):
        solutions.append({
            'title': 'Sleep Improvement',
            'description': 'Improve sleep hygiene: fixed bedtimes, no screens 1hr before sleep, cool dark room. Consider melatonin if needed.',
            'icon': 'bi-moon-stars',
            'priority': 2
        })

    # Study load
    study = str(form_data.get('study_load', '')).lower()
    if study in ('high', '5'):
        solutions.append({
            'title': 'Study Planning',
            'description': 'Break tasks into smaller chunks. Use Pomodoro (25min work, 5min break). Prioritize with a to-do list.',
            'icon': 'bi-journal-text',
            'priority': 2
        })

    # Career concerns
    career = form_data.get('future_career_concerns', 3)
    try:
        c = int(career)
        if c >= 4:
            solutions.append({
                'title': 'Career Guidance',
                'description': 'Visit campus career center. Set short-term goals. Explore internships to build confidence.',
                'icon': 'bi-briefcase',
                'priority': 2
            })
    except (ValueError, TypeError):
        pass

    # Social support
    social = str(form_data.get('social_support', '')).lower()
    if social in ('low', '1'):
        solutions.append({
            'title': 'Build Social Connections',
            'description': 'Join clubs, study groups, or online communities. Schedule regular calls with family/friends.',
            'icon': 'bi-people',
            'priority': 2
        })

    # Bullying
    bullying = str(form_data.get('bullying', '')).lower()
    if bullying in ('often', 'sometimes', '5', '3'):
        solutions.append({
            'title': 'Address Bullying',
            'description': 'Report to school counselor or trusted adult. Document incidents. Seek support groups.',
            'icon': 'bi-shield-check',
            'priority': 1
        })

    # Mental health history
    mh = str(form_data.get('mental_health_history', '')).lower()
    if mh in ('yes', '1'):
        solutions.append({
            'title': 'Professional Support',
            'description': 'Consider regular counseling or therapy. University health services often offer free sessions.',
            'icon': 'bi-person-hearts',
            'priority': 1
        })

    # Medium/High stress: add counseling
    if stress_level_label in ('Medium', 'High'):
        solutions.append({
            'title': 'Counseling Suggestion',
            'description': f'Given your {stress_level_label} stress level, speaking with a counselor can provide personalized coping strategies.',
            'icon': 'bi-chat-dots',
            'priority': 1
        })

    # High stress: emphasize professional help
    if stress_level_label == 'High':
        solutions.append({
            'title': 'Reach Out for Help',
            'description': 'Please contact campus counseling or a mental health professional. You don\'t have to face this alone.',
            'icon': 'bi-telephone',
            'priority': 0
        })

    # Sort by priority (lower = more important)
    solutions.sort(key=lambda x: x['priority'])
    return [{'title': s['title'], 'description': s['description'], 'icon': s['icon']} for s in solutions]
