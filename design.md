# Personal DESIGN.md

This file is the single source of truth for my visual style. It is optimized for low ambiguity and low token cost when used by agents.

## 1. Source Lock

Use these sources exactly:

| Facet | Source |
|---|---|
| Color Palette | Apple + Notion |
| Typography | Claude |
| Button Variants | Apple + Notion |
| Cards | Notion |
| Forms | Apple |
| Spacing | Notion |
| Radius | Apple |
| Elevation | Notion |

## 2. Conflict Resolution

If two systems disagree, resolve in this order:

1. Use the source assigned to that facet.
2. If a component touches multiple facets, split by property:
   - color -> Color Palette
   - font / size / line-height / letter-spacing -> Typography
   - padding / margins / gaps -> Spacing
   - border-radius -> Radius
   - shadow / border-depth treatment -> Elevation
3. Do not import Claude colors, Apple typography, or Notion radius unless this file explicitly allows it.

## 3. Non-Negotiable Rules

- Headlines use Claude typography only: `Anthropic Serif`, weight `500`.
- Body/UI text uses Claude typography only: `Anthropic Sans`.
- Code/monospace uses Claude typography only: `Anthropic Mono`.
- The only primary saturated UI accent is Apple Blue `#0071e3`.
- Notion colors are allowed only as neutrals, semantic colors, and badge colors.
- Default card/button radius is Apple `8px`.
- Default input radius is Apple `11px`.
- Default full pill radius is Apple `980px`.
- Default elevation uses Notion layered shadows, not Apple single-shadow depth.
- No decorative gradients in UI chrome.
- No glassmorphism outside explicit Apple nav patterns.
- No purple-first palettes.
- Serif is for headings and editorial pull quotes only. Never use serif in buttons, forms, nav, or dense UI.

## 4. Core Tokens

### 4.1 Color Tokens

```yaml
color:
  bg.page: "#ffffff"
  bg.section.warm: "#f6f5f4"
  bg.section.cool: "#f5f5f7"
  bg.page.dark: "#000000"
  bg.card: "#ffffff"
  bg.card.warm: "#f6f5f4"
  bg.input: "#fafafc"

  text.primary: "#1d1d1f"
  text.primary.dark: "#ffffff"
  text.secondary: "#615d59"
  text.muted: "#a39e98"
  text.soft: "rgba(0,0,0,0.8)"
  text.tertiary: "rgba(0,0,0,0.48)"

  border.default: "rgba(0,0,0,0.1)"
  border.soft: "#dddddd"

  accent.primary: "#0071e3"
  accent.link: "#0066cc"
  accent.link.dark: "#2997ff"
  accent.focus: "#0071e3"

  badge.info.bg: "#f2f9ff"
  badge.info.text: "#097fe8"

  semantic.success: "#1aae39"
  semantic.success.alt: "#2a9d99"
  semantic.warning: "#dd5b00"
```

### 4.2 Typography Tokens

```yaml
font:
  heading: '"Anthropic Serif", Georgia, serif'
  body: '"Anthropic Sans", Arial, system-ui, sans-serif'
  mono: '"Anthropic Mono", Arial, monospace'

type:
  hero:
    family: heading
    size: 64px
    weight: 500
    lineHeight: 1.10
    letterSpacing: normal
  section:
    family: heading
    size: 52px
    weight: 500
    lineHeight: 1.20
    letterSpacing: normal
  h3:
    family: heading
    size: 32px
    weight: 500
    lineHeight: 1.10
    letterSpacing: normal
  cardTitle:
    family: heading
    size: 20.8px
    weight: 500
    lineHeight: 1.20
    letterSpacing: normal
  bodyLarge:
    family: body
    size: 20px
    weight: 400
    lineHeight: 1.60
    letterSpacing: normal
  body:
    family: body
    size: 16px
    weight: 400
    lineHeight: 1.60
    letterSpacing: normal
  bodyStrong:
    family: body
    size: 16px
    weight: 500
    lineHeight: 1.60
    letterSpacing: normal
  button:
    family: body
    size: 16px
    weight: 500
    lineHeight: 1.25
    letterSpacing: normal
  label:
    family: body
    size: 12px
    weight: 500
    lineHeight: 1.25
    letterSpacing: 0.12px
  overline:
    family: body
    size: 10px
    weight: 400
    lineHeight: 1.60
    letterSpacing: 0.5px
```

### 4.3 Spacing Tokens

```yaml
spacing:
  base: 8px
  scale: [2px, 3px, 4px, 5px, 6px, 7px, 8px, 11px, 12px, 14px, 16px, 24px, 32px]
  container.max: 1200px
  section.y.desktop.min: 80px
  section.y.desktop.max: 120px
  section.y.mobile: 48px
  card.padding.default: 24px
  card.padding.large: 32px
  grid.gap.default: 24px
  grid.gap.tight: 16px
```

### 4.4 Radius Tokens

```yaml
radius:
  micro: 5px
  standard: 8px
  comfortable: 11px
  large: 12px
  pill: 980px
  circle: 50%
```

### 4.5 Elevation Tokens

```yaml
elevation:
  border.whisper: "1px solid rgba(0,0,0,0.1)"
  shadow.card: "rgba(0,0,0,0.04) 0px 4px 18px, rgba(0,0,0,0.027) 0px 2.025px 7.84688px, rgba(0,0,0,0.02) 0px 0.8px 2.925px, rgba(0,0,0,0.01) 0px 0.175px 1.04062px"
  shadow.deep: "rgba(0,0,0,0.01) 0px 1px 3px, rgba(0,0,0,0.02) 0px 3px 7px, rgba(0,0,0,0.02) 0px 7px 15px, rgba(0,0,0,0.04) 0px 14px 28px, rgba(0,0,0,0.05) 0px 23px 52px"
  focus.ring: "2px solid #0071e3"
```

## 5. Component Specs

### 5.1 Buttons

Use Claude typography for all button text.

| Variant | Background | Text | Border | Padding | Radius | Shadow | Hover | Active | Focus |
|---|---|---|---|---|---|---|---|---|---|
| Primary Solid | `#0071e3` | `#ffffff` | `1px solid transparent` | `8px 15px` | `8px` | none | slightly brighten bg | darken 6% | `2px solid #0071e3` |
| Secondary Neutral | `rgba(0,0,0,0.05)` | `#1d1d1f` | `1px solid transparent` | `8px 16px` | `8px` | none | text darkens, bg darkens 4% | scale `0.98` | `2px solid #0071e3` |
| Pill Outline Link | `transparent` | `#0066cc` | `1px solid #0066cc` | `8px 15px` | `980px` | none | underline | darken text 6% | `2px solid #0071e3` |
| Badge Pill | `#f2f9ff` | `#097fe8` | `1px solid transparent` | `4px 8px` | `980px` | none | bg darkens 3% | scale `0.98` | `2px solid #0071e3` |

Button usage:

- Use `Primary Solid` for the main CTA only.
- Use `Secondary Neutral` for secondary actions.
- Use `Pill Outline Link` for "Learn more", "Details", "Open", "View".
- Use `Badge Pill` only for status, labels, "New", "Beta", "Updated".

### 5.2 Cards

Cards inherit Notion structure, Apple radius, Claude type, Notion elevation.

```yaml
card:
  background: "#ffffff"
  background.alt: "#f6f5f4"
  border: "1px solid rgba(0,0,0,0.1)"
  radius.default: 8px
  radius.featured: 12px
  shadow.default: "rgba(0,0,0,0.04) 0px 4px 18px, rgba(0,0,0,0.027) 0px 2.025px 7.84688px, rgba(0,0,0,0.02) 0px 0.8px 2.925px, rgba(0,0,0,0.01) 0px 0.175px 1.04062px"
  padding.default: 24px
  padding.large: 32px
  title:
    family: heading
    size: 20.8px
    weight: 500
    lineHeight: 1.20
  body:
    family: body
    size: 16px
    weight: 400
    lineHeight: 1.60
    color: "#615d59"
```

Card rules:

- Default card background is `#ffffff`.
- Alternate card background is `#f6f5f4`.
- Use featured cards at `12px` radius only when the card is a hero/spotlight block.
- Standard card image corners: top-left/top-right `8px`, bottom corners `0`.
- Card hover may only intensify shadow. Do not add large motion.

### 5.3 Forms

Forms inherit Apple control styling, Claude type, Apple radius, Apple focus color.

```yaml
form:
  field.background: "#fafafc"
  field.text: "rgba(0,0,0,0.8)"
  field.placeholder: "rgba(0,0,0,0.48)"
  field.border: "3px solid rgba(0,0,0,0.04)"
  field.radius: 11px
  field.padding.x: 14px
  field.padding.y: 10px
  field.minHeight: 44px
  field.shadow: none
  field.focus: "2px solid #0071e3"
  label:
    family: body
    size: 12px
    weight: 500
    lineHeight: 1.25
    letterSpacing: 0.12px
    color: "#1d1d1f"
  helpText:
    family: body
    size: 14px
    weight: 400
    lineHeight: 1.43
    color: "#615d59"
```

Form rules:

- Inputs, selects, search fields, and textareas use the same shell.
- Use `44px` minimum control height.
- Textareas use `14px` horizontal padding and `12px` vertical padding.
- Do not use heavy borders, inset shadows, or rounded pills for forms.

## 6. Layout Rules

### 6.1 Web

```yaml
web:
  container.max: 1200px
  content.centered: true
  section.paddingY.desktop: [80px, 120px]
  section.paddingY.mobile: 48px
  grid.columns.desktop: [1, 2, 3]
  grid.gap: 24px
  nav.height: 48px
```

Web rules:

- Default page background is `#ffffff`.
- Alternate sections use `#f6f5f4` first, `#f5f5f7` second, and `#000000` only for emphasis sections.
- Use one product/object/idea hero per section.
- Keep layouts centered and calm. No noisy mosaics.

### 6.2 Images / Posters / Social Graphics

```yaml
image:
  aspectRatios: ["1:1", "4:5", "16:9"]
  safeMargin: 48px
  headline.maxWidth: "10-12 words"
  body.maxWidth: "24-40 words"
  preferredBackgrounds: ["#ffffff", "#f6f5f4", "#f5f5f7", "#000000"]
  preferredRadius: 8px
```

Image rules:

- One focal object only.
- One headline block only.
- Use serif headlines on quiet backgrounds.
- Use Apple Blue only for CTA chips, links, or one tiny accent object.
- If a card frame appears, use Notion card border + Notion shadow + Apple `8px` radius.

### 6.3 PPT / Keynote / Slides

```yaml
slide:
  ratio: "16:9"
  safeMargin: 64px
  title.size: 52px
  title.weight: 500
  title.lineHeight: 1.20
  sectionTitle.size: 64px
  body.size: 20px
  body.lineHeight: 1.60
  caption.size: 14px
  grid.gap: 24px
```

Slide rules:

- Title slides may use `64px` hero headings.
- Content slides use `52px` section titles and `20px` body text.
- Maximum 3 content blocks per slide.
- Use white or warm-white backgrounds for most slides.
- Use black slides only for section breaks or premium emphasis.
- Use cards for data clusters, quotes, and comparisons.

## 7. Prohibitions

Do not generate any of the following unless explicitly requested:

- generic SaaS gradients
- purple-glow dashboards
- neumorphism
- glass cards over every surface
- oversized 16px+ radius everywhere
- heavy drop shadows with opacity above `0.05`
- all-sans typography
- cold blue-gray neutral systems
- more than one saturated CTA color

## 8. Derived Normalization Note

Most tokens above are taken directly from the referenced Apple, Notion, and Claude DESIGN.md files. The following values are explicit normalization decisions added here so agents can work with fewer follow-up questions:

- `form.field.padding.y: 10px`
- `form.field.minHeight: 44px`
- `web.nav.height: 48px`
- all image/poster safe-margin rules
- all slide safe-margin and slide-scale rules

These derived values must still obey the locked source style in Section 1.

## 9. Agent Prompt Shortcut

If an agent needs a minimal instruction, use this exact summary:

```text
Use Apple + Notion colors, Claude typography, Apple + Notion buttons, Notion cards, Apple forms, Notion spacing, Apple radius, and Notion elevation. Primary accent is #0071e3 only. Headings use Anthropic Serif 500. Body/UI use Anthropic Sans. Base spacing is 8px. Default card/button radius is 8px, input radius is 11px, pill radius is 980px. Default border is 1px solid rgba(0,0,0,0.1). Default card shadow is the Notion 4-layer shadow stack. No gradients, no purple-first UI, no generic SaaS look.
```

## 10. Source References

- Apple DESIGN.md: https://raw.githubusercontent.com/VoltAgent/awesome-design-md/main/design-md/apple/DESIGN.md
- Notion DESIGN.md: https://raw.githubusercontent.com/VoltAgent/awesome-design-md/main/design-md/notion/DESIGN.md
- Claude DESIGN.md: https://raw.githubusercontent.com/VoltAgent/awesome-design-md/main/design-md/claude/DESIGN.md
