try:
    import playwright
    print("PLAYWRIGHT_INSTALLED")
except ImportError:
    print("NO_PLAYWRIGHT")

try:
    import selenium
    print("SELENIUM_INSTALLED")
except ImportError:
    print("NO_SELENIUM")
