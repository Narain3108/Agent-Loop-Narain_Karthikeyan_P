from reportlab.pdfgen import canvas

def create_sample_pdf():
    c = canvas.Canvas("sample.pdf")
    # Page 1
    c.drawString(100, 750, "Agentic loops are a new paradigm in AI engineering.")
    c.drawString(100, 730, "They allow an LLM to iteratively perceive its environment,")
    c.drawString(100, 710, "reason about what to do next, take actions using tools,")
    c.drawString(100, 690, "and reflect on the outcome.")
    c.showPage()
    
    # Page 2
    c.drawString(100, 750, "The perception stage structures raw data into an observation.")
    c.drawString(100, 730, "The reasoning stage acts as the brain, formulating a plan.")
    c.drawString(100, 710, "The action stage executes the plan by interacting with the world.")
    c.showPage()

    # Page 3
    c.drawString(100, 750, "Finally, the reflection stage evaluates if the goal was met.")
    c.drawString(100, 730, "If not, it provides feedback to the reasoning stage to try again.")
    c.drawString(100, 710, "This process continues until the task is complete or max iterations reached.")
    c.save()
    print("Created sample.pdf")

if __name__ == '__main__':
    create_sample_pdf()
