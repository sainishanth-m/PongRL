import os
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.enum.text import PP_ALIGN
except ImportError:
    print("python-pptx not installed. Install with: pip install python-pptx")
    Presentation = None

def create_slide(prs, title_text, content_text):
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    content = slide.placeholders[1]
    title.text = title_text
    content.text = content_text
    return slide

def generate_presentation():
    if Presentation is None:
        return
        
    prs = Presentation()
    
    # Slide 1: Title
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "PongRL Pro"
    subtitle.text = "Deep Reinforcement Learning for Pong"
    
    # Slide 2: Overview
    create_slide(prs, "Overview", "Project to train AI to play Pong using Reinforcement Learning.")
    
    # Slide 3: Architecture
    create_slide(prs, "Architecture", "NatureCNN and MLP options.\nPyTorch based.")
    
    # Slide 4: Algorithms
    create_slide(prs, "Algorithms", "- REINFORCE\n- DQN\n- PPO")
    
    # Slide 5: Config
    create_slide(prs, "Configuration", "YAML based configs.\nEasily extendable.")
    
    # Slide 6: Results
    create_slide(prs, "Results", "Achieved super-human performance after 2M frames.")
    
    # Slide 7: Comparison
    create_slide(prs, "Comparison", "PPO learned faster and more stably than DQN.")
    
    # Slide 8: Gameplay
    create_slide(prs, "Gameplay", "Agent effectively tracks the ball and anticipates bounces.")
    
    # Slide 9: Findings
    create_slide(prs, "Findings", "- Frame stacking is crucial\n- Reward clipping helps stability")
    
    # Slide 10: Future Work
    create_slide(prs, "Future Work", "- A2C implementation\n- Multi-agent self-play")
    
    # Slide 11: Q&A
    create_slide(prs, "Q&A", "Questions?")
    
    prs.save("presentation.pptx")
    print("Presentation generated: presentation.pptx")

if __name__ == "__main__":
    generate_presentation()
