---
title: ""
draft: false
showToc: false
summary: "About me."
---

{{< rawhtml >}}

<style>
.about-container {
  display: flex;
  align-items: flex-start; /* align top edges of image+caption and text */
  gap: 40px;               /* space between image and text */
}

.profile-card {
  flex: 0 0 250px;         /* fix width for left column */
  text-align: center;
}

.profile-pic {
  width: 250px;
  aspect-ratio: 1 / 1;
  border-radius: 50%;
  object-fit: cover;
  display: block;
  margin: 0 auto;
}

.caption h3 {
  margin: 12px 0 6px;
  font-weight: 700;
  font-size: 22px;
}
.caption p {
  margin: 0;
  color: #555;
}

.text-column {
  flex: 1;                 /* right column takes remaining space */
}

@media (max-width: 700px) {
  .about-container {
    flex-direction: column;
    align-items: center;
    text-align: center;
  }
  .text-column {
    text-align: left;      /* keep text normal */
  }
}
</style>

<div class="about-container">
  <div class="profile-card">
    <img src="/parth.png" alt="Profile picture" class="profile-pic">
    <div class="caption">
      <h3>Parth Badgujar</h3>
        <small>B.Tech. in Electronics and Communication Engineering at IIT Roorkee</small>
    </div>
  </div>

  <div class="text-column">
{{< /rawhtml >}}

Hi, I am Parth
  

I'm final year undergrad at IIT Roorkee majoring in <a href="https://ece.iitr.ac.in">Electronics and Communication Engineering</a>. My interests span `AI Security`, `Diffusion Models`, `ML Systems` and `Systems Security`. 
 
I am associated with {{<rawhtml>}}<a href="https://dsgiitr.in/" style="color: blue;">Data Science Group</a>{{</rawhtml>}} at IIT Roorkee, where we build projects and research at the frontier of AI. 
 
Having an electronics background I enjoy fiddling with low level systems with hands on experience in optimizing <code>CUDA</code> / <code>Triton</code> kernels.
 
I am an active CTF player with {{<rawhtml>}} <a href="https://ctftime.org/team/16691/" style="color: blue;">InfoSecIITR</a> {{</rawhtml>}} (CTF team from IIT Roorkee) where I work on binary and linux kernel exploitation.  

Apart from my main quests I enjoy studying about rockets and airplanes.

{{< rawhtml >}}
  </div>
</div>
{{< /rawhtml >}}
