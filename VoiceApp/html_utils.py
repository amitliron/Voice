import settings

def convert_text_to_html(full_html, current_text, current_lang):
    '''
        style text results in html format
    '''

    if current_lang == "he" or current_lang == "ar":
        current_line = f"<p style='text-align:right;'> {current_text} </p>"
    else:
        current_line = f"<p style='text-align:left;'> {current_text} </p>"

    full_html.append(current_line)
    return full_html

def build_html_table(all_results):

    html_table = []
    html_table.append("<table border='1'  align='center'>")
    for text, lang in all_results:
        if lang == "he":
            html_table.append(f"<tr align='right'>")
        else:
            html_table.append(f"<tr align='left'>")
        html_table.append(f"<td> {text}</td>")
        html_table.append(f"<td> {settings.LANGUAGES[lang]} </td>")
        html_table.append("</tr>")
    html_table.append("</table>")

    return html_table