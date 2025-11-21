import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.units import inch
import diagnostics


with open('config.json', 'r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


def generate_pdf_report():
    # Create PDF file
    pdf_file = os.path.join(model_path, 'model_report.pdf')
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    
    # Container for PDF elements
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph("ML Model Performance Report", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 0.3 * inch))
    
    # Timestamp
    timestamp = Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles['Normal']
    )
    elements.append(timestamp)
    elements.append(Spacer(1, 0.3 * inch))
    
    # F1 Score
    score_file = os.path.join(prod_deployment_path, 'latestscore.txt')
    with open(score_file, 'r') as f:
        f1_score = f.read()
    
    score_text = Paragraph(f"<b>F1 Score:</b> {f1_score}", styles['Heading2'])
    elements.append(score_text)
    elements.append(Spacer(1, 0.2 * inch))
    
    # Summary Statistics
    stats = diagnostics.dataframe_summary()
    stats_title = Paragraph("<b>Summary Statistics</b>", styles['Heading2'])
    elements.append(stats_title)
    elements.append(Spacer(1, 0.1 * inch))
    
    # Create statistics table
    stats_data = [
        ['Metric', 'lastmonth_activity', 'lastyear_activity', 'number_of_employees'],
        ['Mean', f'{stats[0]:.2f}', f'{stats[3]:.2f}', f'{stats[6]:.2f}'],
        ['Median', f'{stats[1]:.2f}', f'{stats[4]:.2f}', f'{stats[7]:.2f}'],
        ['Std Dev', f'{stats[2]:.2f}', f'{stats[5]:.2f}', f'{stats[8]:.2f}']
    ]
    
    stats_table = Table(stats_data)
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(stats_table)
    elements.append(Spacer(1, 0.3 * inch))
    
    # Execution Times
    timings = diagnostics.execution_time()
    timing_title = Paragraph("<b>Execution Times</b>", styles['Heading2'])
    elements.append(timing_title)
    elements.append(Spacer(1, 0.1 * inch))
    
    timing_text = Paragraph(
        f"Ingestion: {timings[0]:.4f}s | Training: {timings[1]:.4f}s",
        styles['Normal']
    )
    elements.append(timing_text)
    elements.append(Spacer(1, 0.3 * inch))
    
    # Confusion Matrix
    cm_file = os.path.join(model_path, 'confusionmatrix.png')
    if os.path.exists(cm_file):
        cm_title = Paragraph("<b>Confusion Matrix</b>", styles['Heading2'])
        elements.append(cm_title)
        elements.append(Spacer(1, 0.1 * inch))
        
        cm_image = Image(cm_file, width=4*inch, height=3*inch)
        elements.append(cm_image)
    
    # Build PDF
    doc.build(elements)
    print(f"PDF report generated: {pdf_file}")


if __name__ == '__main__':
    generate_pdf_report()
