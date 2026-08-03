print("Testing Import FrontEnd & Classification Model")

print("-----------------------------------------")
# FrontEnd
print("Testing FrontEnd Model")
try:
    from FrontEnd import Leaf
    print("FrontEnd : Leaf is Working")
except:
    print("FrontEnd : Leaf fail to load.")

try:
    from FrontEnd import Mel
    print("FrontEnd : Mel is Working")
except:
    print("FrontEnd : Mel fail to load.")

try:
    from FrontEnd import Scatter
    print("FrontEnd : Scatter is Working")
except:
    print("FrontEnd : Scatter fail to load.")

print("-----------------------------------------")
# Model
print("Testing Classification Model")
try:
    from Model import VitCnnGlobal
    print("Classification Model : VitCnnGlobal is Working")
except:
    print("Classification Model : VitCnnGlobal fail to load.")

try:
    from Model import VitCnnLocal
    print("Classification Model : VitCnnLocal is Working")
except:
    print("Classification Model : VitCnnLocal fail to load.")

try:
    from Model import VitGlobal
    print("Classification Model : VitGlobal is Working")
except:
    print("Classification Model : VitGlobal fail to load.")

try:
    from Model import VitLocal
    print("Classification Model : VitLocal is Working")
except:
    print("Classification Model : VitLocal fail to load.")

print("-----------------------------------------")