from main import match_user_response
import unittest

class TestSynonymsMatching(unittest.TestCase):
    def test_ios_matching(self):
        ios_tests = [
            "айфон", 
            "я хочу разрабатывать для iPhone", 
            "для айфона пожалуйста", 
            "apple устройства", 
            "на айфоне и айпаде"
            "говнофон"
        ]
        
        for test in ios_tests:
            result, score, *_ = match_user_response(test, ["ios", "android"], "platform")
            self.assertEqual(result, "ios", f"Тест не прошел для '{test}', получили {result} вместо 'ios'")
    
    def test_android_matching(self):
        android_tests = [
            "андроид", 
            "разработка для андройда", 
            "гугл устройства", 
            "самсунг и другие", 
            "на андроиде и планшетах"
        ]
        
        for test in android_tests:
            result, score, *_ = match_user_response(test, ["ios", "android"], "platform")
            self.assertEqual(result, "android", f"Тест не прошел для '{test}', получили {result} вместо 'android'")

if __name__ == "__main__":
    unittest.main()