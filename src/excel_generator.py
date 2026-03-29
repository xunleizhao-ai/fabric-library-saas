import xlsxwriter
import os
import cv2

def generate_fabric_excel(processed_data_list, output_filename="AI_Standardized_Fabric_Library.xlsx"):
    """
    Generates an Excel file with extracted fabric data and embedded swatch images.

    Args:
        processed_data_list (list): A list of dictionaries, where each dictionary contains:
                                    - 'Brand' (str)
                                    - 'Item' (str)
                                    - 'Content' (str)
                                    - 'swatch_image_path' (str): Path to the saved swatch image.
        output_filename (str): The name of the Excel file to create.

    Returns:
        str: The path to the generated Excel file.
    """
    workbook = xlsxwriter.Workbook(output_filename)
    worksheet = workbook.add_worksheet()

    # Define formats
    header_format = workbook.add_format({'bold': True, 'bg_color': '#D9EAD3', 'border': 1})
    text_wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'vcenter'})

    # Headers (consistent with user's Colab and requirements)
    headers = ['Brand Name', 'Item Number', 'Fabric Content', 'Swatch Image']
    for col_num, data_head in enumerate(headers):
        worksheet.write(0, col_num, data_head, header_format)

    # Set column widths
    worksheet.set_column('A:B', 20)
    worksheet.set_column('C:C', 35)
    worksheet.set_column('D:D', 30)

    row = 1
    for data_entry in processed_data_list:
        worksheet.set_row(row, 150) # Set row height for image

        worksheet.write(row, 0, data_entry.get('Brand', ''), text_wrap_format)
        worksheet.write(row, 1, data_entry.get('Item', ''), text_wrap_format)
        worksheet.write(row, 2, data_entry.get('Content', ''), text_wrap_format)
        
        # Insert image if path exists
        image_path = data_entry.get('swatch_image_path')
        if image_path and os.path.exists(image_path):
            worksheet.insert_image(row, 3, image_path, {'x_scale': 0.4, 'y_scale': 0.4})
        
        row += 1
    
    workbook.close()
    return output_filename


def clean_up_temp_images(processed_data_list):
    """
    Deletes temporary swatch image files after they have been embedded in the Excel.
    """
    for data_entry in processed_data_list:
        image_path = data_entry.get('swatch_image_path')
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
            # print(f"Cleaned up: {image_path}") # For debugging
