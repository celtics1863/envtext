def _wash_label(label):
    return label.strip() \
             .replace('  ','') \
             .lower()

def _is_english_char(ch):
    if 97 <= ord(ch) <= 122 or 65 <= ord(ch)<= 90:
        return True
    else:
        return False
    
def _is_english_char_lower(ch):
    if 97 <= ord(ch) <= 122:
        return True
    else:
        return False

def _is_english_char_upper(ch):
    if 65 <= ord(ch)<= 90:
        return True
    else:
        return False