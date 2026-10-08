import re

html_path = 'D:/fetal-health-classification-main/frontend/templates/index.html'
with open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()

# I will find the broken Portal 1 and remove it completely cleanly
# Currently it looks like:
#           <!-- Portal 1: 50 Patients -->
#           
#             <div class="quick-card-info">
#               <div class="quick-card-title">50 Patients <span class="quick-arrow">&gt;</span></div>
#               <div class="quick-card-sub">MATCHED SYNTHETIC CASES</div>
#             </div>
#           </div>

html = html.replace('''          <!-- Portal 1: 50 Patients -->
          
            <div class="quick-card-info">
              <div class="quick-card-title">50 Patients <span class="quick-arrow">&gt;</span></div>
              <div class="quick-card-sub">MATCHED SYNTHETIC CASES</div>
            </div>
          </div>''', '')

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)
print("Fixed Portal 1 HTML")
