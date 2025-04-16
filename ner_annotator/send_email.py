import smtplib
from email.message import EmailMessage

def send_json_via_gmail(
    sender_email: str,
    sender_password: str,
    recipient_email: str,
    subject: str,
    body: str,
    json_filepath: str
):
    # Create the email
    msg = EmailMessage()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.set_content(body)

    # Attach the JSON file
    with open(json_filepath, 'rb') as f:
        file_data = f.read()
        file_name = json_filepath.split('/')[-1]
    msg.add_attachment(
        file_data,
        maintype='application',
        subtype='json',
        filename=file_name
    )

    # Send it off
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)
        print(f"Email sent to {recipient_email} with attachment {file_name}!")
