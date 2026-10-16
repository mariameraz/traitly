---
hide:
  - navigation
  - toc

---

<style>

.md-typeset .grid.cards > ul {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important; /* two columns */
    gap: 1rem;
    align-items: stretch;
}

/* flexible cards */
.md-typeset .grid.cards > ul > li {
    display: flex;
    flex-direction: column;
    margin: 0 !important;
    box-sizing: border-box;
}

/* Image ratio 16:9 */
.md-typeset .grid.cards img {
    width: 100%;
    aspect-ratio: 16 / 9;
    object-fit: contain;
    display: block;
}

.md-typeset .grid.cards > ul > li > p:last-child {
    margin-top: auto;
}

.md-typeset .grid.cards .card-title-big {
    display: block;
    text-align: center;
    font-weight: 600;
}

/* Reduce cols to 1 in small windows */
@media (max-width: 800px) {
    .md-typeset .grid.cards > ul {
        grid-template-columns: 1fr !important;
    }
}
</style>


<div class="animate" markdown>

# Gallery ⊹ ࣪ ˖

Here you will find examples of images analyzed with Traitly. Click on a title to see the code behind each image.

---

<br>

</div>

<div class="grid cards" markdown>

-   ![Beans](https://i.swiftimg.com/variants/r4JRIOum/display.webp)

    ---

    [Bean morphological analysis](scripts/bean.md){ .card-title-big }

-   ![Gehler dataset](https://i.swiftimg.com/variants/GpMXfYll/display.webp)

    ---

    [Corrected image from the Gehler dataset](scripts/color_correction.md){ .card-title-big }


-   ![Strawberry color analysis](https://i.swiftimg.com/variants/q1pzqApz/display.webp)

    ---

    [Strawberry color analysis](scripts/strawberry.md){ .card-title-big }


-   ![Potato](https://i.swiftimg.com/originals/1Yo0vcOW.png)

    ---

    [Potato morphological analysis](scripts/potato.md){ .card-title-big }

-   ![Cucumber](https://i.swiftimg.com/variants/lofO2pit/display.webp)

    ---

    [Cucumber internal fruit analysis](scripts/cucumber.md){ .card-title-big }


-   ![Gehler dataset 2](https://i.swiftimg.com/variants/H3V1qXRj/display.webp)

    ---

    [Corrected image from the Gehler dataset](scripts/color_correction_photo2.md){ .card-title-big }

</div>
