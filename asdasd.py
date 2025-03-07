import random
import streamlit as st
import streamlit.components.v1 as components

def content():
    for i in range(50):
        st.write(f"line {i}: {random.randint(0, 1000)}")


html_code = f"""
<div id="scroll-to-me" style='background: cyan; height=1px;'>hi</div>
<script id="{random.randint(1000, 9999)}">
   var e = document.getElementById("scroll-to-me");
   if (e) {{
     e.scrollIntoView({{behavior: "smooth"}});
     e.remove();
   }}
</script>
"""

st.subheader("Content")
if st.button("Scroll to bottom"):
    content()
    components.html(html_code)
else:
    content()