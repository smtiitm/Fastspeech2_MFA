class TextReplacer:
    def __init__(self):
        self.replacements = {
        'aa': 'A',
        'ii': 'I',
        'uu': 'U',
        'ee': 'E',
        'oo': 'O',
        'nn': 'N',
        'ae': 'ऍ',
        'ag': 'ऽ',
        'au': 'औ',
        'axx': 'अ',
        'ax': 'ऑ',
        'bh': 'B',
        'ch': 'C',
        'dh': 'ध',
        'dx': 'ड',
        'dxh': 'ढ',
        'dxhq': 'T',
        'dxq': 'D',
        'ei': 'ऐ',
        'ai': 'ऐ',
        'eu': 'உ',
        'gh': 'घ',
        'gq': 'G',
        'hq': 'H',
        'jh': 'J',
        'kh': 'ख',
        'khq': 'K',
        'kq': 'क',
    # ... Add more replacements as needed
        }
    
    def apply_replacements(self, text):
        for key, value in self.replacements.items():
            # print('KEY AND VALUE OF PARSED OUTPUT',key, value)
            text = text.replace(key, value)
            text = text.replace(" ", "")
            
        
        return text

