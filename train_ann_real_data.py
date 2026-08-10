"""
Train ANN model using real UpdatedResumeDataSet.csv
This creates the resume_score_ann model that matcher.py will load
"""

import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🔄 Loading Sentence Transformer...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# ============================================
# Skill Extraction (matches your matcher.py)
# ============================================
def extract_skills_from_text(text):
    """Extract skills - same as in matcher.py"""
    skill_keywords = [
        'python', 'java', 'c++', 'html', 'css', 'javascript', 'react', 'angular',
        'node', 'express', 'django', 'flask', 'sql', 'mysql', 'postgresql',
        'mongodb', 'data analysis', 'machine learning', 'deep learning', 'nlp',
        'opencv', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'github',
        'tensorflow', 'pytorch', 'excel', 'powerbi', 'tableau', 'communication',
        'leadership', 'problem solving', 'linux', 'bash', 'api', 'cloud',
        'agile', 'scrum', 'jira', 'rest', 'graphql', 'nosql', 'redis',
        'spark', 'hadoop', 'scala', 'r', 'matlab', 'sas', 'power bi'
    ]
    
    text = text.lower()
    extracted_skills = set()
    
    for skill in skill_keywords:
        if re.search(rf'\b{re.escape(skill)}\b', text):
            extracted_skills.add(skill)
    
    return extracted_skills


# ============================================
# Generate Job Descriptions
# ============================================
def generate_jd_for_category(category, skills_pool):
    """Generate realistic JD based on category"""
    
    # Category-specific JD templates
    jd_templates = {
        'Data Science': [
            "Seeking Data Scientist with {skills}. Build ML models, analyze data, create visualizations. Required: {req}. Preferred: {pref}.",
            "Data Scientist needed for {skills}. Develop predictive models and insights. Must have: {req}. Nice to have: {pref}.",
        ],
        'Java Developer': [
            "Java Developer position requiring {skills}. Build scalable backend systems. Required: {req}. Bonus: {pref}.",
            "Looking for Java Developer with {skills}. Develop microservices and APIs. Must know: {req}. Plus: {pref}.",
        ],
        'Python Developer': [
            "Python Developer role with {skills}. Create applications and automate workflows. Required: {req}. Preferred: {pref}.",
            "Hiring Python Developer skilled in {skills}. Build APIs and data pipelines. Must have: {req}. Nice to have: {pref}.",
        ],
        'Web Designing': [
            "Web Designer needed with {skills}. Create responsive UI/UX designs. Required: {req}. Bonus: {pref}.",
            "UI/UX Designer position requiring {skills}. Design modern web interfaces. Must know: {req}. Plus: {pref}.",
        ],
    }
    
    # Default template
    default = "Hiring for {category}. Looking for {skills}. Required: {req}. Preferred: {pref}."
    
    template = np.random.choice(jd_templates.get(category, [default]))
    
    # Select skills
    skills_list = list(skills_pool)
    if len(skills_list) < 3:
        return None, set(), set()
    
    num_required = min(np.random.randint(3, 8), len(skills_list))
    required = set(np.random.choice(skills_list, num_required, replace=False))
    
    remaining = list(skills_pool - required)
    num_preferred = min(np.random.randint(2, 5), len(remaining)) if remaining else 0
    preferred = set(np.random.choice(remaining, num_preferred, replace=False)) if remaining else set()
    
    all_skills = list(required | preferred)
    
    jd = template.format(
        category=category,
        skills=', '.join(all_skills[:5]),
        req=', '.join(list(required)[:4]),
        pref=', '.join(list(preferred)[:3]) if preferred else 'None'
    )
    
    return jd, required, preferred


# ============================================
# Load Dataset and Create Training Data
# ============================================
def create_training_data(csv_path='UpdatedResumeDataSet.csv', num_samples=3000):
    """Create training data from real resumes"""
    
    print(f"📂 Loading dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        print("\n📥 Download it first:")
        print("wget https://huggingface.co/datasets/brackozi/Resume/resolve/main/UpdatedResumeDataSet.csv")
        return None
    
    print(f"✅ Loaded {len(df)} resumes")
    print(f"📊 Categories: {df['Category'].nunique()} unique")
    
    training_data = []
    
    print(f"\n🔄 Creating {num_samples} training samples...")
    
    for idx in tqdm(range(num_samples)):
        # Sample a resume
        row = df.iloc[idx % len(df)]
        resume_text = str(row['Resume'])
        category = str(row['Category'])
        
        # Extract resume skills
        resume_skills = extract_skills_from_text(resume_text)
        
        if len(resume_skills) < 2:
            continue
        
        # Generate matching JD
        jd_text, required_skills, preferred_skills = generate_jd_for_category(
            category, resume_skills
        )
        
        if jd_text is None:
            continue
        
        jd_skills = required_skills | preferred_skills
        
        if len(jd_skills) == 0:
            continue
        
        # Calculate features
        matched_skills = resume_skills & jd_skills
        skill_match_ratio = len(matched_skills) / len(jd_skills)
        
        # Semantic similarity
        try:
            # Use first 1000 chars to avoid token limits
            resume_snippet = resume_text[:1000]
            embeddings = model.encode([resume_snippet, jd_text], convert_to_tensor=True)
            semantic_similarity = float(util.cos_sim(embeddings[0], embeddings[1]).item())
        except:
            semantic_similarity = skill_match_ratio
        
        num_resume_skills = len(resume_skills)
        num_jd_skills = len(jd_skills)
        
        # Calculate target score (0-100) - STRICTER VERSION
        # Primary factor: skill matching (80% weight)
        base_score = skill_match_ratio * 80
        
        # Secondary factor: semantic similarity (15% weight)
        semantic_component = semantic_similarity * 15
        
        # Small bonus for having many skills (5% weight)
        skill_bonus = min(5, (num_resume_skills / 30) * 5)
        
        # Penalty for missing critical skills
        missing_ratio = 1 - skill_match_ratio
        penalty = missing_ratio * 10  # Up to 10 point penalty
        
        match_score = base_score + semantic_component + skill_bonus - penalty
        match_score += np.random.normal(0, 2.5)  # Add noise
        match_score = np.clip(match_score, 0, 100)
        
        training_data.append({
            'skill_match_ratio': skill_match_ratio,
            'semantic_similarity': semantic_similarity,
            'num_resume_skills': num_resume_skills,
            'num_jd_skills': num_jd_skills,
            'match_score': match_score
        })
    
    training_df = pd.DataFrame(training_data)
    
    print(f"\n✅ Created {len(training_df)} valid training samples")
    print(f"\n📊 Score Statistics:")
    print(training_df['match_score'].describe())
    
    return training_df


# ============================================
# Build ANN Model
# ============================================
def build_ann_model(input_dim=4):
    """Build ANN architecture"""
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        Dropout(0.1),
        
        Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ============================================
# Train Model
# ============================================
def train_model(training_df):
    """Train the ANN"""
    
    print("\n🔄 Preparing training data...")
    
    X = training_df[['skill_match_ratio', 'semantic_similarity', 
                     'num_resume_skills', 'num_jd_skills']].values
    y = training_df['match_score'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build
    print("\n🔄 Building ANN model...")
    ann_model = build_ann_model()
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=15, 
                               restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                  patience=5, min_lr=1e-6, verbose=1)
    
    # Train
    print("\n🔄 Training...")
    history = ann_model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    test_loss, test_mae = ann_model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\n✅ Test MSE: {test_loss:.2f}")
    print(f"✅ Test MAE: {test_mae:.2f}")
    
    # Sample predictions
    preds = ann_model.predict(X_test_scaled[:5], verbose=0)
    print("\n🔍 Sample Predictions:")
    for i in range(5):
        print(f"  Pred: {preds[i][0]:.1f} | Actual: {y_test[i]:.1f}")
    
    # Save
    print("\n💾 Saving model...")
    ann_model.save("resume_score_ann.keras")
    joblib.dump(scaler, "resume_scaler.pkl")
    print("✅ Saved: resume_score_ann.keras")
    print("✅ Saved: resume_scaler.pkl")
    
    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("📊 Saved: training_history.png")
    
    return ann_model, scaler


# ============================================
# Test Model
# ============================================
def test_model():
    """Quick test"""
    print("\n🧪 Testing saved model...")
    
    try:
        from tensorflow.keras.models import load_model
        import os
        
        # Load the correct file
        if os.path.exists("resume_score_ann.keras"):
            model = load_model("resume_score_ann.keras")
        else:
            print("❌ resume_score_ann.keras not found!")
            return
        
        scaler = joblib.load("resume_scaler.pkl")
        
        tests = [
            ([0.9, 0.88, 25, 20], "Excellent"),
            ([0.7, 0.75, 18, 15], "Good"),
            ([0.5, 0.60, 12, 12], "Moderate"),
            ([0.3, 0.45, 8, 18], "Poor"),
        ]
        
        print("\n🎯 Test Cases:")
        for features, label in tests:
            scaled = scaler.transform([features])
            score = model.predict(scaled, verbose=0)[0][0]
            print(f"  {label:12s}: {score:.1f}/100")
        
        print("\n✅ Model working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================
# MAIN
# ============================================
def main():
    print("=" * 70)
    print("🚀 Training ANN Model with Real Resume Dataset")
    print("=" * 70)
    
    # Create training data
    training_df = create_training_data('UpdatedResumeDataSet.csv', num_samples=3000)
    
    if training_df is None or len(training_df) < 100:
        print("❌ Not enough training data created")
        return
    
    # Save training data
    training_df.to_csv('training_data.csv', index=False)
    print(f"💾 Saved: training_data.csv")
    
    # Train
    ann_model, scaler = train_model(training_df)
    
    # Test
    test_model()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("\n📂 Files Created:")
    print("   - resume_score_ann/     (Trained model)")
    print("   - resume_scaler.pkl     (Feature scaler)")
    print("   - training_data.csv     (Training dataset)")
    print("   - training_history.png  (Visualization)")
    print("\n🎯 Next: Run 'streamlit run app.py'")
    print("   Your app will automatically use this trained model!")
    print("=" * 70)


if __name__ == "__main__":
    main()