# PULSE - Emotional Wellness Application

A comprehensive emotional wellness application that helps users understand their emotions, manage tasks, and enhance their well-being through AI-powered analysis.

## Features

- **User Registration**: Multi-step onboarding process to collect user profile information
- **AI-Generated Questions**: Personalized emotional profiling questions using Gemini API
- **Emotion Recognition**: 3x3 grid of emotion images for intuitive emotion selection
- **AI Analysis**: Comprehensive emotional analysis with personalized suggestions
- **Task Management**: AI-recommended tasks based on emotional state
- **Productivity Reminders**: Smart reminders to help users stay productive
- **Visual Results**: Color-coded emotional state representation

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript, Tailwind CSS
- **AI**: Google Gemini API for question generation and analysis
- **ML**: PyTorch for emotion recognition (optional)
- **Styling**: Tailwind CSS with custom design system

## Installation

1. **Clone the repository** (if applicable) or ensure you have all files in the correct directory structure

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the dataset**:
   - Place your emotion image dataset in: `E:\ML\Projects\datasets\archive\archive (3)\Train`
   - The dataset should have subfolders for each emotion: amusement, anger, awe, contentment, disgust, excitement, fear, sadness

4. **Configure API Key**:
   - The Gemini API key is already configured in `app.py`
   - If you need to change it, update the `GEMINI_API_KEY` variable

## Running the Application

### Method 1: Using the run script
```bash
python run.py
```

### Method 2: Direct Flask execution
```bash
python app.py
```

### Method 3: Flask CLI
```bash
export FLASK_APP=app.py
flask run
```

The application will start on `http://localhost:5000`

## Application Flow

1. **Welcome Page** (`/`) - Introduction to PULSE
2. **Registration Pages** (`/page/2`, `/page/3`, `/page/4`) - User profile collection
3. **Questions** (`/questions.html`) - AI-generated personalized questions
4. **Emotion Grid** (`/emotion-grid.html`) - Visual emotion selection
5. **Results** (`/results.html`) - AI analysis and recommendations

## API Endpoints

- `GET /` - Welcome page
- `POST /register` - User registration/profile updates
- `GET /generate-questions` - Generate personalized questions
- `GET /get-emotion-grid` - Get emotion images for selection
- `POST /analyze-emotions` - Analyze user emotions and provide recommendations
- `GET /page/<int:page_num>` - Serve registration pages
- `GET /questions.html` - Questions page
- `GET /emotion-grid.html` - Emotion selection page
- `GET /results.html` - Results page

## File Structure

```
pulse-app/
├── app.py                 # Main Flask application
├── run.py                 # Application runner script
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── ui/                   # Frontend files
    ├── page1.html        # Welcome page
    ├── page2.html        # Basic info registration
    ├── page3.html        # Lifestyle info
    ├── page4.html        # Goals and health
    ├── questions.html    # AI-generated questions
    ├── emotion-grid.html # Emotion selection
    ├── results.html      # Analysis results
    └── ...               # Other existing pages
```

## Configuration

### Dataset Path
Update the `DATASET_PATH` in `app.py` to point to your emotion image dataset:

```python
DATASET_PATH = "path/to/your/emotion/dataset/Train"
```

### Model Path
Update the `MODEL_PATH` in `app.py` if you have a trained emotion recognition model:

```python
MODEL_PATH = "path/to/your/emotion_model.pth"
```

### Gemini API Key
Update the `GEMINI_API_KEY` in `app.py`:

```python
GEMINI_API_KEY = "your_gemini_api_key_here"
```

## Features in Detail

### User Registration
- Multi-step form collection
- Personal information, lifestyle habits, goals
- Health issue tracking
- Relationship and occupation details

### AI Question Generation
- Uses user profile to generate personalized questions
- Multiple choice format (A, B, C, D options)
- Focuses on emotional patterns and behavioral insights

### Emotion Recognition
- Visual emotion selection through image grid
- 9 random images from 8 emotion categories
- Intuitive click-to-select interface

### AI Analysis
- Combines user profile, selected emotions, and answers
- Generates personalized suggestions
- Creates task recommendations with priorities
- Provides productivity reminders
- Calculates emotional balance score

### Results Display
- Color-coded emotional state representation
- Personalized suggestions and tasks
- Productivity statistics
- Action buttons for new sessions or insights

## Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure the emotion dataset is in the correct path
2. **API Key issues**: Verify your Gemini API key is valid and has proper permissions
3. **Port already in use**: Change the port in `app.run(port=5001)`
4. **Model loading errors**: The app will work without the PyTorch model (uses fallback)

### Debug Mode
The application runs in debug mode by default. For production:
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and personal use. Please ensure compliance with API usage policies and data privacy regulations.

## Support

For issues or questions, please check the troubleshooting section or create an issue in the repository.
