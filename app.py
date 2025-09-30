from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import random
import requests
import re
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import base64
import io

app = Flask(__name__, static_folder='ui', template_folder='ui')

# Configuration
GEMINI_API_KEY = "AIzaSyCzIh-kejDyiczqiJwe6cSkh18fp4Rn0F0"
MODEL_PATH = "E:\ML\Projects\INFERENTIA\emotion_efficientnet_best_pytorch.pth"
DATASET_PATH = r"C:\Users\dell\Desktop\projects\inferentia\New folder\image\all images"

# Global variables
user_data = {}
current_questions = []
selected_emotions = []

# Emotion classes
EMOTIONS = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

class EmotionModel(nn.Module):
    def __init__(self, num_classes=8):
        super(EmotionModel, self).__init__()
        # EfficientNet backbone would be loaded here
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load model
try:
    model = EmotionModel()
    # model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_random_emotion_images():
    """Get 9 random images from the dataset directory where emotions are labeled in filenames"""
    images = []
    selected_emotions = []

    try:
        if not os.path.exists(DATASET_PATH):
            print(f"Dataset path does not exist: {DATASET_PATH}")
            return [], []

        # Get all JPG files from the directory
        all_images = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith('.jpg')]
        print(f"Found {len(all_images)} JPG images in dataset")

        if not all_images:
            print("No JPG images found in dataset directory")
            return [], []

        # Group images by emotion based on filename
        emotion_images = {}
        for emotion in EMOTIONS:
            emotion_images[emotion] = []

        for img_file in all_images:
            img_name = img_file.lower()
            for emotion in EMOTIONS:
                if emotion in img_name:
                    emotion_images[emotion].append(img_file)
                    break

        # Select one image per emotion if available
        for emotion in EMOTIONS:
            if emotion_images[emotion]:
                selected_img = random.choice(emotion_images[emotion])
                images.append({
                    'path': os.path.join(DATASET_PATH, selected_img),
                    'emotion': emotion
                })
                selected_emotions.append(emotion)

        # If we don't have 9 images, fill with available ones
        while len(images) < 9 and images:
            remaining_emotions = [e for e in EMOTIONS if e not in selected_emotions]
            if remaining_emotions:
                emotion = random.choice(remaining_emotions)
                if emotion_images[emotion]:
                    selected_img = random.choice(emotion_images[emotion])
                    images.append({
                        'path': os.path.join(DATASET_PATH, selected_img),
                        'emotion': emotion
                    })
                    selected_emotions.append(emotion)
            else:
                # If no more unique emotions, duplicate existing ones
                break

        print(f"Selected {len(images)} images for emotions: {selected_emotions}")
        return images[:9], selected_emotions[:9]

    except Exception as e:
        print(f"Error in get_random_emotion_images: {e}")
        return [], []

def call_gemini_api(prompt, system_prompt=""):
    """Call Gemini API with the given prompt"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    payload = {
        "contents": [{
            "parts": [{"text": full_prompt}]
        }]
    }

    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

@app.route('/')
def index():
    return render_template('page1.html')

@app.route('/page/<int:page_num>')
def serve_page(page_num):
    return render_template(f'page{page_num}.html')

@app.route('/questions.html')
def questions_page():
    return render_template('questions.html')

@app.route('/emotion-grid.html')
@app.route('/emotion-grid')
def emotion_grid_page():
    return render_template('emotion-grid.html')

@app.route('/results.html')
def results_page():
    return render_template('results.html')

@app.route('/register', methods=['POST'])
def register_user():
    """Handle user registration"""
    global user_data
    data = request.json

    user_data = {
        'nickname': data.get('nickname'),
        'name': data.get('name'),
        'age': data.get('age'),
        'gender': data.get('gender'),
        'sleep_hours': data.get('sleep_hours'),
        'energetic_time': data.get('energetic_time'),
        'break_time': data.get('break_time'),
        'happy_place': data.get('happy_place'),
        'goals': data.get('goals'),
        'health_issues': data.get('health_issues', []),
        'relationship_status': data.get('relationship_status'),
        'occupation': data.get('occupation'),
        'school_name': data.get('school_name'),
        'major': data.get('major'),
        'timestamp': datetime.now().isoformat()
    }

    return jsonify({'success': True, 'message': 'User registered successfully'})

@app.route('/generate-questions', methods=['GET'])
def generate_questions():
    """Generate 10 profiling questions using Gemini API"""
    global current_questions

    system_prompt = """You are PULSE's profiling assistant.

Your job is to take structured registration answers from the user and generate:

1. A set of 10 intelligent, non-repetitive profiling questions that explore
   the user's lifestyle, personality, habits, values, and preferences.
   These questions should feel natural and empathetic, not robotic.

2. A well-structured profile summary that combines:
   - The original registration responses
   - The answers to your generated questions

3. The final output must be formatted in a professional,
   human-readable style suitable for saving as a PDF.

-------------------------------
Registration Questions Provided:
1. Nickname
2. Full Name
3. Age
4. Gender
   - If female: include period cycle tracking info
5. Hours of sleep (average)
6. Most energetic time of the day
7. Break time & Happy place
8. Assigned colors for 3 basic emotions (captcha style)
9. PULSE goal
10. Health issues (options & free text)
11. Occupation
12. Relationship status
-------------------------------

Instructions:
- After receiving registration data, generate 10 additional questions
  to deepen understanding of the user.
- Incorporate both registration data and these new answers into a
  single coherent profile.
- Write in warm, supportive language (avoid clinical tone).
- Use clear section headers like:
    "Personal Info", "Daily Routine", "Emotional Preferences",
    "Health & Wellness", "Occupation & Lifestyle", "Extra Insights"
- Ensure the text looks good in PDF output (avoid clutter, keep it concise).

Final output = Complete PDF-ready user profile."""

    # Create a more structured prompt for question generation
    user_profile_summary = f"""
User Profile Summary:
- Name: {user_data.get('name', 'Not provided')}
- Age: {user_data.get('age', 'Not provided')}
- Gender: {user_data.get('gender', 'Not provided')}
- Sleep Hours: {user_data.get('sleep_hours', 'Not provided')}
- Most Energetic Time: {user_data.get('energetic_time', 'Not provided')}
- Occupation: {user_data.get('occupation', 'Not provided')}
- Goals: {user_data.get('goals', 'Not provided')}
- Health Issues: {', '.join(user_data.get('health_issues', []))}
"""

    prompt = f"""Based on this user profile, generate exactly 10 multiple-choice questions (MCQ format) that explore the user's lifestyle, personality, habits, values, and preferences. Each question must have exactly 4 options (A, B, C, D).

{user_profile_summary}

Return ONLY a valid JSON array of questions in this exact format:
[
    {{
        "id": 1,
        "question": "How do you typically start your day?",
        "options": ["A) With meditation and planning", "B) With coffee and news", "C) With exercise and music", "D) With family time and breakfast"]
    }},
    {{
        "id": 2,
        "question": "Question text here?",
        "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"]
    }}
]

Make questions empathetic, natural, and insightful for emotional wellness assessment."""

    try:
        response = call_gemini_api(prompt, system_prompt)
        print(f"Gemini API Response: {response}")

        # Clean the response to extract JSON
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]

        response = response.strip()

        # Try to parse JSON response
        try:
            questions_data = json.loads(response)
            if isinstance(questions_data, list):
                current_questions = questions_data
            else:
                current_questions = questions_data.get('questions', [])
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Response was: {response}")
            # If JSON parsing fails, try to extract questions from text response
            current_questions = parse_questions_from_text(response)

        # Ensure we have exactly 10 questions with proper format
        if not current_questions or len(current_questions) < 10:
            print("Using fallback questions")
            current_questions = generate_fallback_questions()

        # Validate and format questions
        formatted_questions = []
        for i, q in enumerate(current_questions[:10]):
            if isinstance(q, dict) and 'question' in q and 'options' in q:
                formatted_questions.append({
                    'id': i + 1,
                    'question': q['question'],
                    'options': q['options'][:4] if len(q['options']) >= 4 else q['options'] + ['Option ' + str(len(q['options']) + j + 1) for j in range(4 - len(q['options']))]
                })

        current_questions = formatted_questions

        return jsonify({'success': True, 'questions': current_questions})

    except Exception as e:
        print(f"Error generating questions: {e}")
        # Fallback: create generic questions
        current_questions = generate_fallback_questions()
        return jsonify({'success': True, 'questions': current_questions})

def parse_questions_from_text(text_response):
    """Parse questions from Gemini's text response"""
    questions = []
    lines = text_response.split('\n')

    current_question = None
    current_options = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if this is a question (starts with number or Q)
        if re.match(r'^\d+\.|\bQ\d*\.|\bQuestion\s*\d*', line, re.IGNORECASE):
            # Save previous question if exists
            if current_question and current_options:
                questions.append({
                    "id": len(questions) + 1,
                    "question": current_question,
                    "options": current_options[:4]  # Take first 4 options
                })

            # Extract question text
            current_question = re.sub(r'^\d+\.|\bQ\d*\.|\bQuestion\s*\d*[:.]?\s*', '', line, flags=re.IGNORECASE).strip()
            current_options = []

        # Check if this is an option (starts with A, B, C, D or letter))
        elif re.match(r'^[A-Da-d][).]\s*|^•\s*', line):
            option_text = re.sub(r'^[A-Da-d][).]\s*|^•\s*', '', line).strip()
            if option_text:
                current_options.append(option_text)

    # Add the last question
    if current_question and current_options:
        questions.append({
            "id": len(questions) + 1,
            "question": current_question,
            "options": current_options[:4]
        })

    return questions

def generate_fallback_questions():
    """Generate fallback questions based on user profile"""
    base_questions = [
        {
            "id": 1,
            "question": "How do you typically feel when starting a new day?",
            "options": ["Energetic and ready to go", "Calm and peaceful", "A bit anxious or stressed", "Tired and unmotivated"]
        },
        {
            "id": 2,
            "question": "When faced with a challenging task, what's your usual approach?",
            "options": ["Break it down into smaller steps", "Seek help from others", "Push through with determination", "Take a break and come back later"]
        },
        {
            "id": 3,
            "question": "How do you handle stress or pressure?",
            "options": ["Practice deep breathing or meditation", "Talk to friends or family", "Exercise or physical activity", "Watch TV or play games to relax"]
        },
        {
            "id": 4,
            "question": "What's your preferred way to spend free time?",
            "options": ["Reading or learning something new", "Socializing with friends", "Creative activities like art or music", "Relaxing with hobbies or entertainment"]
        },
        {
            "id": 5,
            "question": "How do you feel about your daily routine?",
            "options": ["I enjoy the structure and predictability", "I prefer flexibility and spontaneity", "I find it sometimes overwhelming", "I adapt based on my energy levels"]
        },
        {
            "id": 6,
            "question": "What motivates you the most in life?",
            "options": ["Achieving personal goals", "Helping others succeed", "Learning and personal growth", "Maintaining work-life balance"]
        },
        {
            "id": 7,
            "question": "How do you typically react to unexpected changes?",
            "options": ["I adapt quickly and see opportunities", "I feel anxious but manage to cope", "I prefer to plan and avoid surprises", "I get excited about new possibilities"]
        },
        {
            "id": 8,
            "question": "What's your relationship with technology?",
            "options": ["I embrace it and stay updated", "I use it when necessary for work", "I find it overwhelming sometimes", "I enjoy using it for entertainment"]
        },
        {
            "id": 9,
            "question": "How do you feel about social interactions?",
            "options": ["I thrive in social settings", "I prefer smaller, intimate gatherings", "I need alone time to recharge", "I enjoy a mix of both"]
        },
        {
            "id": 10,
            "question": "What does success mean to you?",
            "options": ["Achieving career milestones", "Maintaining healthy relationships", "Personal happiness and fulfillment", "Making a positive impact on others"]
        }
    ]

    return base_questions

@app.route('/get-emotion-grid', methods=['GET'])
def get_emotion_grid():
    """Get 3x3 grid of random emotion images"""
    try:
        print(f"Dataset path: {DATASET_PATH}")
        print(f"Dataset exists: {os.path.exists(DATASET_PATH)}")

        if not os.path.exists(DATASET_PATH):
            # Return fallback images if dataset doesn't exist
            return jsonify({
                'success': True,
                'images': [
                    {
                        'data': 'data:image/svg+xml;base64,' + base64.b64encode(f'''
                            <svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
                                <rect width="200" height="200" fill="#f3f4f6"/>
                                <text x="100" y="100" text-anchor="middle" dy=".3em" fill="#6b7280" font-size="14">Image not found</text>
                            </svg>
                        '''.encode()).decode(),
                        'emotion': emotion
                    } for emotion in EMOTIONS[:9]
                ],
                'emotions': EMOTIONS[:9]
            })

        images, emotions = get_random_emotion_images()
        print(f"Found {len(images)} images for emotions: {emotions}")

        # Convert images to base64 for frontend
        image_data = []
        for img_info in images:
            try:
                if os.path.exists(img_info['path']):
                    with open(img_info['path'], 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                        image_data.append({
                            'data': f"data:image/jpeg;base64,{img_data}",
                            'emotion': img_info['emotion']
                        })
                    print(f"Successfully loaded image for {img_info['emotion']}")
                else:
                    print(f"Image file not found: {img_info['path']}")
                    # Add placeholder
                    image_data.append({
                        'data': 'data:image/svg+xml;base64,' + base64.b64encode(f'''
                            <svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
                                <rect width="200" height="200" fill="#f3f4f6"/>
                                <text x="100" y="100" text-anchor="middle" dy=".3em" fill="#6b7280" font-size="12">{img_info['emotion']}</text>
                            </svg>
                        '''.encode()).decode(),
                        'emotion': img_info['emotion']
                    })
            except Exception as e:
                print(f"Error loading image {img_info['path']}: {e}")
                continue

        print(f"Returning {len(image_data)} images")
        return jsonify({
            'success': True,
            'images': image_data,
            'emotions': emotions
        })
    except Exception as e:
        print(f"Error in get_emotion_grid: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze-emotions', methods=['POST'])
def analyze_emotions():
    """Analyze selected emotions and user data using Gemini API"""
    data = request.json
    selected_emotions = data.get('selected_emotions', [])
    question_answers = data.get('question_answers', [])

    # Combine user data with selected emotions and answers
    analysis_data = {
        'user_profile': user_data,
        'selected_emotions': selected_emotions,
        'question_answers': question_answers,
        'timestamp': datetime.now().isoformat()
    }

    system_prompt = """
    You are an emotional wellness AI assistant. Analyze the user's profile, selected emotions, and answers to provide:

    1. A mixed emotion color (hex code) representing their current emotional state
    2. Personalized suggestions for what they should do in their current mental state
    3. Specific tasks they should complete within the closest time frame
    4. Productivity reminders to help them stay on track
    5. Statistics about their emotional patterns

    Consider their:
    - Age, gender, occupation
    - Sleep patterns, energetic times
    - Health issues and goals
    - Selected emotions from the image grid
    - Answers to profiling questions

    Return response in this JSON format:
    {
        "emotion_color": "#HEXCODE",
        "suggestions": ["Suggestion 1", "Suggestion 2", "Suggestion 3"],
        "tasks": [
            {"task": "Task description", "deadline": "time frame", "priority": "high/medium/low"}
        ],
        "reminders": ["Reminder 1", "Reminder 2"],
        "stats": {
            "dominant_emotion": "emotion_name",
            "emotional_balance": "percentage",
            "productivity_score": "score/100"
        }
    }
    """

    prompt = f"Analyze this user's emotional profile: {json.dumps(analysis_data, indent=2)}"

    try:
        response = call_gemini_api(prompt, system_prompt)
        analysis_result = json.loads(response)
        return jsonify({'success': True, 'analysis': analysis_result})
    except Exception as e:
        # Fallback analysis
        fallback_result = {
            "emotion_color": "#A7F3D0",
            "suggestions": [
                "Take a 10-minute walk to clear your mind",
                "Practice deep breathing exercises",
                "Write down three things you're grateful for"
            ],
            "tasks": [
                {"task": "Review daily goals", "deadline": "today", "priority": "high"},
                {"task": "Schedule break time", "deadline": "within 2 hours", "priority": "medium"}
            ],
            "reminders": [
                "Stay hydrated throughout the day",
                "Complete one focused task before checking messages"
            ],
            "stats": {
                "dominant_emotion": selected_emotions[0] if selected_emotions else "neutral",
                "emotional_balance": "75%",
                "productivity_score": "82/100"
            }
        }
        return jsonify({'success': True, 'analysis': fallback_result})

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('ui', filename)

@app.route('/tasks', methods=['GET', 'POST'])
def task_manager():
    """Task manager functionality"""
    if request.method == 'POST':
        task_data = request.json
        # Here you would save tasks to a database
        # For now, we'll just return success
        return jsonify({'success': True, 'message': 'Task added successfully'})

    # Return task list (would come from database in real implementation)
    return jsonify({'success': True, 'tasks': []})

@app.route('/pulse-widget.js')
def pulse_widget():
    """Serve the embeddable PULSE widget"""
    widget_js = '''
(function() {
    // PULSE Widget - Embeddable Emotional Wellness Assistant

    // Configuration
    const WIDGET_CONFIG = {
        position: 'bottom-right',
        size: { width: 340, height: 'auto' },
        theme: 'auto',
        apiEndpoint: window.location.origin
    };

    // Emotion states with colors and content
    const EMOTION_STATES = [
        { name: 'Happy', emoji: '😊', color: '#fbbf24', task: 'Continue your positive momentum!', affirmation: 'You are capable and worthy of joy.' },
        { name: 'Sad', emoji: '😢', color: '#3b82f6', task: 'Take a moment for self-care.', affirmation: 'Your feelings are valid and temporary.' },
        { name: 'Stressed', emoji: '😰', color: '#ef4444', task: 'Try deep breathing exercises.', affirmation: 'You are stronger than your challenges.' },
        { name: 'Motivated', emoji: '💪', color: '#10b981', task: 'Channel your energy into action!', affirmation: 'Your determination creates possibilities.' },
        { name: 'Calm', emoji: '😌', color: '#06b6d4', task: 'Maintain this peaceful state.', affirmation: 'Inner peace is your natural state.' }
    ];

    let currentEmotion = EMOTION_STATES[Math.floor(Math.random() * EMOTION_STATES.length)];
    let widgetContainer = null;
    let isExpanded = false;

    // Create widget styles
    function injectStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .pulse-widget {
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 9999;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }

            .pulse-toggle {
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: linear-gradient(135deg, #8b5cf6, #a855f7);
                border: none;
                cursor: pointer;
                box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4);
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
            }

            .pulse-toggle:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 25px rgba(139, 92, 246, 0.6);
            }

            .pulse-card {
                position: absolute;
                bottom: 80px;
                right: 0;
                width: 340px;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
                border: 1px solid rgba(255, 255, 255, 0.2);
                opacity: 0;
                transform: translateY(20px) scale(0.9);
                transition: all 0.3s ease;
                overflow: hidden;
            }

            .pulse-card.expanded {
                opacity: 1;
                transform: translateY(0) scale(1);
            }

            .pulse-header {
                padding: 20px;
                background: linear-gradient(135deg, #8b5cf6, #a855f7);
                color: white;
                text-align: center;
            }

            .pulse-emoji {
                font-size: 48px;
                margin-bottom: 10px;
            }

            .pulse-emotion-name {
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 5px;
            }

            .pulse-content {
                padding: 20px;
            }

            .pulse-task {
                background: #f8fafc;
                padding: 15px;
                border-radius: 12px;
                margin-bottom: 15px;
                border-left: 4px solid #8b5cf6;
            }

            .pulse-task-title {
                font-weight: bold;
                color: #374151;
                margin-bottom: 5px;
            }

            .pulse-task-text {
                color: #6b7280;
                font-size: 14px;
            }

            .pulse-affirmation {
                background: linear-gradient(135deg, #fef3c7, #fde68a);
                padding: 15px;
                border-radius: 12px;
                margin-bottom: 20px;
                text-align: center;
                font-style: italic;
                color: #92400e;
            }

            .pulse-buttons {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
            }

            .pulse-btn {
                padding: 12px 16px;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s ease;
                text-align: center;
                font-size: 14px;
            }

            .pulse-btn-primary {
                background: linear-gradient(135deg, #8b5cf6, #a855f7);
                color: white;
            }

            .pulse-btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
            }

            .pulse-btn-secondary {
                background: #f3f4f6;
                color: #374151;
            }

            .pulse-btn-secondary:hover {
                background: #e5e7eb;
            }

            @keyframes pulse-glow {
                0%, 100% { box-shadow: 0 0 20px rgba(139, 92, 246, 0.4); }
                50% { box-shadow: 0 0 30px rgba(139, 92, 246, 0.8); }
            }

            .pulse-toggle.pulse {
                animation: pulse-glow 2s infinite;
            }
        `;
        document.head.appendChild(style);
    }

    // Create widget HTML
    function createWidget() {
        widgetContainer = document.createElement('div');
        widgetContainer.className = 'pulse-widget';
        widgetContainer.innerHTML = `
            <button class="pulse-toggle" id="pulseToggle">💡</button>
            <div class="pulse-card" id="pulseCard">
                <div class="pulse-header">
                    <div class="pulse-emoji" id="emotionEmoji">${currentEmotion.emoji}</div>
                    <div class="pulse-emotion-name" id="emotionName">${currentEmotion.name}</div>
                </div>
                <div class="pulse-content">
                    <div class="pulse-task">
                        <div class="pulse-task-title">💡 Suggested Task</div>
                        <div class="pulse-task-text" id="emotionTask">${currentEmotion.task}</div>
                    </div>
                    <div class="pulse-affirmation" id="emotionAffirmation">
                        "${currentEmotion.affirmation}"
                    </div>
                    <div class="pulse-buttons">
                        <button class="pulse-btn pulse-btn-primary" onclick="logMood()">Log Mood</button>
                        <button class="pulse-btn pulse-btn-secondary" onclick="addTask()">+ Task</button>
                        <button class="pulse-btn pulse-btn-secondary" onclick="openPulse()">Open PULSE</button>
                        <button class="pulse-btn pulse-btn-secondary" onclick="changeEmotion()">🔄</button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(widgetContainer);

        // Add event listeners
        const toggleBtn = widgetContainer.querySelector('#pulseToggle');
        const card = widgetContainer.querySelector('#pulseCard');

        toggleBtn.addEventListener('click', function() {
            isExpanded = !isExpanded;
            card.classList.toggle('expanded', isExpanded);

            // Add pulse animation to toggle button
            this.classList.add('pulse');
            setTimeout(() => this.classList.remove('pulse'), 2000);
        });

        // Make functions global for onclick handlers
        window.logMood = function() {
            alert('Mood logged! Keep tracking your emotional wellness.');
            changeEmotion();
        };

        window.addTask = function() {
            alert('Task feature coming soon! Use the full PULSE app for task management.');
        };

        window.openPulse = function() {
            window.open('${WIDGET_CONFIG.apiEndpoint}', '_blank');
        };

        window.changeEmotion = function() {
            const newEmotion = EMOTION_STATES[Math.floor(Math.random() * EMOTION_STATES.length)];
            currentEmotion = newEmotion;

            // Update UI
            document.getElementById('emotionEmoji').textContent = currentEmotion.emoji;
            document.getElementById('emotionName').textContent = currentEmotion.name;
            document.getElementById('emotionTask').textContent = currentEmotion.task;
            document.getElementById('emotionAffirmation').textContent = `"${currentEmotion.affirmation}"`;

            // Update card background color
            card.style.borderColor = currentEmotion.color + '40';
        };
    }

    // Initialize widget
    function init() {
        injectStyles();
        createWidget();

        // Auto-change emotion every 30 seconds when expanded
        setInterval(() => {
            if (isExpanded) {
                changeEmotion();
            }
        }, 30000);
    }

    // Start when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
'''
    return widget_js, 200, {'Content-Type': 'application/javascript'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
