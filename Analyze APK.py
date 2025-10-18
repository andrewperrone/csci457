import os
from androguard.misc import AnalyzeAPK

apk_root_folder = "test"  # or whatever the root folder for apks is called

if not os.path.exists(apk_root_folder):
    print(f"Error: Folder '{apk_root_folder}' does not exist!")
    exit()

#set only holds unique values, no duplicates
unique_permissions = set()
unique_api_calls = set()

#iterate through all apks in the folder and subfolders
for root, dirs, files in os.walk(apk_root_folder):
    for apk_file in files:
        if apk_file.endswith(".apk"): #only care about .apk files
            apk_path = os.path.join(root, apk_file) #full path to the apk
            try:
                #analyze the apk
                a, d, dx = AnalyzeAPK(apk_path)
                # Collect permissions 
                unique_permissions.update(a.get_permissions())
                # Collect API calls
                for method in dx.get_methods():
                    #iterate through all calls in the method
                    for _, call, _ in method.get_xref_to():
                        #code readability
                        classname = call.class_name[1:-1].replace(
                            "/", ".")  # Remove leading 'L' and trailing ';'
                        methodname = call.name
                        if classname.startswith(
                                "android.") or classname.startswith("java."):
                            unique_api_calls.add(f"{classname}.{methodname}")
            except Exception as e:
                print(f"Error processing {apk_file}: {e}")
with open("unique_permissions.txt", "w", encoding="utf-8") as perm_file:
    for perm in sorted(unique_permissions):
        perm_file.write(f"{perm}\n")
# Save unique API calls to a text file, one per line
with open("unique_api_calls.txt", "w", encoding="utf-8") as api_file:
    for api in sorted(unique_api_calls):
        api_file.write(f"{api}\n")
print("Extraction complete. Unique permissions and API calls saved.")