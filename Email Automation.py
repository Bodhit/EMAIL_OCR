import os
import glob
import re
import cv2
import pytesseract
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import time
import numpy as np
from IPython.display import display, Image

# --- Configuration ---
GMAIL_ADDRESS = "xyz@gmail.com"  # Your Gmail address
APP_PASSWORD = ""  # Gmail App Password/Secret Key (I have personally used key)
SCREENSHOT_DIR = "XX/Test_emails"  # Directory with screenshots
RESUME_PATH = "XXResume - DataScience.pdf"  # Path to resume
OUTPUT_CSV = os.path.join(SCREENSHOT_DIR, "extracted_emails.csv")  # CSV output path
EMAIL_SUBJECT = "Application for Data Scientist Role – Bodhit Tangri"
EMAIL_BODY = """Dear Hiring Manager,

I hope this message finds you well.

My name is Bodhit Tangri, and I am writing to express my keen interest in data science opportunities within your organization. With over four years of experience in risk analytics and decision science at American Express and a strong foundation in statistical modeling, machine learning, and anomaly detection, I believe I bring a valuable combination of technical expertise and strategic thinking.

Currently, I serve as a Senior Analyst in Risk Management, where I have led the development of a Trade-Based Money Laundering framework that increased true positive alert rates by 60%, and engineered detection techniques that improved fraud identification by 30%. My work consistently demonstrates an ability to derive actionable insights from complex data, automate processes, and build scalable solutions—all critical skills for a high-impact data scientist.

I am proficient in Python, SQL, PySpark, and BigQuery, and I hold certifications in Machine Learning and Advanced Data Analytics from Coursera. I’m particularly drawn to roles that merge data science with real-world problem-solving in financial or technology-driven environments.

I have attached my resume for your review and would welcome the opportunity to further discuss how I can contribute to your team. Thank you for considering my application—I look forward to the possibility of connecting.

Warm regards,
Bodhit Tangri
📞 +91-91XXX
📧 XXXX
🔗 https://www.linkedin.com/in/{Your email address}
"""

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# --- Set Tesseract Path (macOS) ---
tesseract_path = "/opt/homebrew/bin/tesseract"  # Apple Silicon
if not os.path.exists(tesseract_path):
    tesseract_path = "/usr/local/bin/tesseract"  # Intel Mac fallback
    if not os.path.exists(tesseract_path):
        raise RuntimeError("Tesseract not found. Install with `brew install tesseract`.")
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# --- Image Preprocessing ---
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"File does not exist: {image_path}")
        return None
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological closing to clean up text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Resize for better OCR
    resized = cv2.resize(closed, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    
    return resized

# --- Split Image into Rows ---
def split_image_into_rows(image, row_height=60):
    h, w = image.shape
    rows = []
    for i in range(0, h, row_height):
        row = image[i:min(i + row_height, h), :]
        rows.append(row)
    return rows

# --- Extract Emails ---
def extract_emails_from_screenshot(image_path, row_height=60):
    try:
        processed_img = preprocess_image(image_path)
        if processed_img is None:
            return []
        display(Image(filename=image_path))
        
        # Split into rows
        rows = split_image_into_rows(processed_img, row_height)
        
        emails = []
        for idx, row in enumerate(rows):
            text = pytesseract.image_to_string(row, config='--psm 6 --oem 1')
            print(f"Extracted text from row {idx}:\n{text}")
            email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
            row_emails = re.findall(email_pattern, text)
            valid_emails = [email for email in row_emails if re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.(com|org|net|edu|gov)$", email)]
            emails.extend(valid_emails)
        
        print(f"Extracted {len(emails)} valid emails from {image_path}: {emails[:5]}...")
        return list(set(emails))
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []

# --- Process All Screenshots ---
def get_all_emails():
    email_addresses = []
    screenshot_files = glob.glob(os.path.join(SCREENSHOT_DIR, "*.png")) + \
                      glob.glob(os.path.join(SCREENSHOT_DIR, "*.jpg"))
    
    print(f"Found {len(screenshot_files)} image files: {screenshot_files}")
    for screenshot in screenshot_files:
        print(f"Processing {screenshot}...")
        emails = extract_emails_from_screenshot(screenshot)
        email_addresses.extend(emails)
    
    email_addresses = list(set(email_addresses))
    
    if os.path.exists(OUTPUT_CSV):
        backup_csv = os.path.join(SCREENSHOT_DIR, f"extracted_emails_backup_{time.strftime('%Y%m%d_%H%M')}.csv")
        os.rename(OUTPUT_CSV, backup_csv)
        print(f"Backed up existing CSV to {backup_csv}")
    
    if email_addresses:
        pd.DataFrame(email_addresses, columns=["email"]).to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {len(email_addresses)} emails to {OUTPUT_CSV}")
    else:
        print("No emails extracted!")
    return email_addresses

# --- Send Email ---
def send_email(to, subject, body, resume_path):
    try:
        msg = MIMEMultipart()
        msg["From"] = GMAIL_ADDRESS
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        
        if os.path.exists(resume_path):
            with open(resume_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(resume_path)}")
                msg.attach(part)
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(GMAIL_ADDRESS, APP_PASSWORD)
            server.send_message(msg)
            print(f"Email sent to {to}")
    except Exception as e:
        print(f"Error sending email to {to}: {e}")

# --- Main Execution ---
def main():
    print("Extracting email addresses from screenshots...")
    email_addresses = get_all_emails()
    print(f"Found {len(email_addresses)} unique email addresses:")
    if email_addresses:
        display(pd.DataFrame(email_addresses, columns=["email"]))
        
        input("Press Enter to start sending emails (review extracted emails in CSV)...")
        print("Sending emails...")
        for i, email in enumerate(email_addresses, 1):
            send_email(email, EMAIL_SUBJECT, EMAIL_BODY, RESUME_PATH)
            time.sleep(0.5)
            if i % 10 == 0:
                print(f"Processed {i}/{len(email_addresses)} emails")
        print("Email sending complete!")
    else:
        print("No emails to send.")

if __name__ == "__main__":
    main()