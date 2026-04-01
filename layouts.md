# A compilation of information about the layout formats usable in this app.

There are two distinct formats:
* the _simple_ format is a text file with one key label per line, and two consecutive line breaks indicating a switch to the next row, [example](app/src/main/assets/layouts/main/qwerty.txt)
* the _json_ format taken from [FlorisBoard](https://github.com/florisboard/florisboard/blob/master/CONTRIBUTING.md#adding-the-layout), but only "normal" keys are supported (i.e. no action keys and similar), [example](app/src/main/assets/layouts/main/azerty.json)

You can add both directly in the app, see the related [Wiki page](https://github.com/HeliBorg/HeliBoard/wiki/2.-Layouts).

## General notes
Adding too many keys or too long texts will make the keyboard look awkward or broken, and even crash the app under some specific conditions (popup keys are especially prone for this).
There are some sanity checks when adding a layout to avoid such issues, but they do not cover all possible cases.
Further there is no check whether the layout actually contains characters of the selected language.

If you use an external glide typing library, you likely will have issues if your layout contains duplicate keys, or keys with text longer than a single character.

If the layout has exactly 2 keys in the bottom row, these keys will replace comma and period keys. More exactly: the first key will replace the first functional key with `"groupId": 1` in the bottom row, and the second key with replace the first key with `"groupId": 2`.

## Simple format
* One key per line
  * Key format: [label] [popup keys], all separated by space, e.g. `a 0 + *` will create a key with text `a`, and the keys `0`, `+`, and `*` on long press
  * see [below](#labels) for information about special labels
* Two consecutive newlines mark beginning of a new row

## Json format
* Normal json layout with [lenient](https://kotlinlang.org/api/kotlinx.serialization/kotlinx-serialization-json/kotlinx.serialization.json/-json-builder/is-lenient.html) parsing, and ignoring lines starting with `//`.
  * For anything else than small changes and copy/pasting text the in-app editor is unsuitable. A proper text editor (e.g. Kate or Notepad++) can significantly simplify work on json files.
* Allows more flexibility than the simple format, e.g. changing keys depending on input type, shift state or layout direction
* You can use character layouts from [FlorisBoard](https://github.com/florisboard/florisboard/blob/master/CONTRIBUTING.md#adding-the-layout)
  * Support is not 100% there yet, notably `kana_selector` and `char_width_selector` do not work.
* There is no need for specifying a `code`, it will be determined from the label automatically
  * You can still specify it, but it's only necessary if you want key label and code to be different (please avoid contributing layout with unnecessary codes to HeliBoard)
  * Note that not all _special codes_ (negative numbers) from FlorisBoard are supported
* Key classes: specified with `$`, usually you can omit them in HeliBoard 
  * `text_key`: normal key, default
  * `auto_text_key`: used in FlorisBoard for a key that changes text case when shift is enabled, HeliBoard does that anyway unless disabled with a _labelFlag_
  * `multi_text_key`: key with an array of code points, e.g. `{ "$": "multi_text_key", "codePoints": [2509, 2480], "label": "্র" }`
  * there are also selector classes, which allow to change keys conditionally, see the [dvorak layout](app/src/main/assets/layouts/main/dvorak.json) for an example:
    * `case_selector`: keys for `lower` and `upper` (both mandatory), similar to `shift_state_selector`
    * `shift_state_selector`: keys for `unshifted`, `shifted`, `shiftedManual`, `shiftedAutomatic`, `capsLock`, `manualOrLocked`, `default` (all optional)
    * `variation_selector`: keys for input types `datetime`, `time`, `date`, `password`, `normal`, `uri`, `email`, `default` (all optional)
    * `keyboard_state_selector`: keys for `emojiKeyEnabled`, `languageKeyEnabled`, `symbols`, `moreSymbols`, `alphabet`, `default` (all optional)
      * the `<emoji/language>KeyEnabled` keys will be used if the corresponding setting is enabled, `symbols`, `moreSymbols`, `alphabet` will be used when the said keyboard view is active
    * `layout_direction_selector`: keys for `ltr` and `rtl` (both mandatory)
### Properties
* A (non-selector) key can have the following properties:
* `type`: only specific values, HeliBoard mostly uses this to determine background color and type, determined automatically by default
  * `normal`: normal key color
  * `function`: functional key color
  * `space`: space bar color
  * `action`: action key color
  * `unspecified`: no background color
  * `placeholder`: no background color, no label, and pressing the key does nothing
  * `numeric`: normal key color, only in number layouts: sets default width to `-1` and sets default label flags if none specified
  * There are some more values, but they do nothing
* `code`: code point that is entered when the key is pressed, determined from the label by default, not available for `multi_text_key`
  * There are special negative values available, e.g. the ones used by functional keys, see [KeyCode.kt](/app/src/main/java/helium314/keyboard/keyboard/internal/keyboard_parser/floris/KeyCode.kt). There are several not yet supported key codes in there, you can see in the function `checkAndConvertCode` which ones are working.
  * Notes for the modifier keys `CTRL`, `ALT`, `FN`, `META` and the left/right/lock versions
    * The lock versions (`CTRL_LOCK`, `ALT_LOCK`, `FN_LOCK`, `META_LOCK`) will stay active until pressed again. The normal versions will stay active until released or a code input happens (whatever comes later). Avoid having both lock and non-lock versions of the same key on a keyboard, they interact badly.
    * Some applications will only react to a specific `_LEFT` or `_RIGHT` version of a meta key
* `codePoints`: when multiple code points should be entered, only available for `multi_text_key`
* `label`: text to display on the key, determined from code if empty
  * There are some special values, see the [label section](#labels)
* `groupId`: which additional popup keys to show, `0` is default and does not add anything, `1` adds the comma popup keys, `2` adds the period popup keys, `3` adds the action key popup keys (looks awkward though), `-1` suppresses additional popups based on the label
* `popup`: list of keys to add in the popup, e.g. `"label": ")", "popup": {"relevant": [{  "label": "." }]}` is a `)` key with a `.` popup
  * Note that in popup keys, properties are ignored with the exception of `$`, `code`, `codePoints`, and `label`
  * When specifying a _selector_ key class in a popup key, it will be evaluated correctly (e.g. for changing popups dependent on shift state)
  * If popups are added to repeating keys (e.g. delete, arrow keys), repetition will be disabled.
* `width`: width of the key in units of screen width, e.g. a key with `"width": 0.1` has a width of 10% of the screen, defaults to `0`
  * A special value is `-1`, which means the key expands to the available space not already used by other keys (e.g. the space bar)
  * `0` is interpreted as follows
    * `-1` on the `space` key in alphabet or symbols layouts, and for keys with `"type": numeric` in number layouts
    * `0.17` for number layouts
    * `0.1` for phones
    * `0.09` for tablets
  * If the sum of widths in a row is greater than 1, keys are rescaled to fit on the screen
* `labelFlags`: allows specific effects, see [here](app/src/main/res/values/attrs.xml#L250-L282) in the section _keyLabelFlags_ for names and numeric values
  * Since json does not support hexadecimal-values, you have to use the decimal values in the comments in the same line.
  * In case you want to apply multiple flags, you will need to combine them using [bitwise OR](https://en.wikipedia.org/wiki/Bitwise_operation#OR). In most cases this means you can just add the individual values, only exceptions are `fontDefault`, `followKeyLabelRatio`, `followKeyHintLabelRatio`, and `autoScale`.

## Labels
In the simple format you only specify labels, in json layouts you do it explicitly via the `label` property.
Usually the label is what is displayed on the key. However, there are some special labels:
* Currency keys
  * `$$$` will be replaced by the local currency, depending on your current layout language. If you define a key with `$$$` without defining popup keys, it will get the first 4 additional currencies (see below) as popup
  * `$$$1` - `$$$5` will be replaced by currencies available on long-pressing the currency key
* Functional keys (incomplete list)
  * _alpha_: switch to alphabet keyboard (or main phone keyboard in case of phone layout)
  * _symbol_: switch to symbol keyboard (or phone symbols keyboard in case of phone layout)
  * _symbol_alpha_: toggle alpha / symbol keyboard
  * _numpad_: toggle numpad layout
  * _emoji_: switch to emoji view
  * _com_: display common TLDs (.com and similar, localized)
  * _language_switch_: language switch key
  * _action_: the action (enter) key
  * _delete_: delete key
  * _shift_: shift key, will change label when in symbols layout
  * _period_: `.` key with punctuation popups, will adapt to language-specific period
  * _comma_: `,` key with special popups, will adapt to language-specific comma, or display `/` in URL fields and `@` in email fields
  * _space_: space key, with icon when using a number layout
  * _zwnj_: Zero-width non-joiner (automatically added next to space in alphabet layout for some languages)
  * You can also use [toolbar keys](/app/src/main/java/helium314/keyboard/latin/utils/ToolbarUtils.kt#L109), e.g. _undo_.
  * See [KeyLabel.kt](app/src/main/java/helium314/keyboard/keyboard/internal/keyboard_parser/floris/KeyLabel.kt) for more available labels that are parsed to the corresponding key.
* In case a label clashes with text you want to add, put a `\` in front of the text you want, e.g. `\space` will write the label `space` instead of adding a space bar.
  * Note that you need to escape the `\` in json files by adding a second `\`.
* If you want different key label and input text, set the label to [label]|[text], e.g. `aa|bb` will show `aa`, but pressing the key will input `bb`.
You can also specify special key codes like `a|!code/key_action_previous` or `abc|!code/-10043`, but it's cleaner to use a json layout and specify the code explicitly. Note that when specifying a code in the label, and a code in a json layout, the code in the label will be ignored.
* It's also possible to specify an icon, like `!icon/previous_key|!code/key_action_previous`.
  * You can find available icon names in [KeyboardIconsSet](/app/src/main/java/helium314/keyboard/keyboard/internal/KeyboardIconsSet.kt). You can also use toolbar key icons using the uppercase name of the [toolbar key](/app/src/main/java/helium314/keyboard/latin/utils/ToolbarUtils.kt#L109), e.g. `!icon/redo`
* There are some further special labels to be used in popup keys (i.e. one of the popup keys should have the label)
  * `!noPanelAutoPopupKey!`: no popups are shown, a long press will result in the first normal popup of the key being selected
  * `!needsDividers!`: dividers are shown between popup keys
  * `!hasLabels!`: reduces text size in popup keys for nicer display of labels instead of letters
  * `!autoColumnOrder!`: use with a number, e.g. _!autoColumnOrder!4_ will result in 4 popup columns
  * `!fixedColumnOrder!`: use with a number, e.g. _!fixedColumnOrder!4_ will result in 4 popup columns. Keys will not be re-ordered if the result is a single line.

## Adding new layouts / languages
* You need a layout file in one of the formats above, and add it to [layouts](app/src/main/assets/layouts)
  * Popup keys in the layout will be in the "_Layout_" popup key group.
  * If you add a json layout, only add key type (`$`) and `code` if necessary
* Add a layout entry to [`method.xml`](app/src/main/res/xml/method.xml)
  * `KeyboardLayoutSet` in `android:imeSubtypeExtraValue` must be set to the name of your layout file (without file ending)
  * `android:subtypeId` must be set to a value that is unique in this file (please use the same length as for other layouts)
  * If you add a layout to an existing language, add a string with the layout name to use instead of `subtype_generic`. The new string should be added to default [`strings.xml`](/app/src/main/res/values/strings.xml), and optionally to other languages. `%s` will be replaced with the language.
* If you add a new language, you might want to provide a [locale_key_texts](/app/src/main/assets/locale_key_texts) file
  * `[popup_keys]` section contains popup keys that are similar to the letter (like `a` and `ä` or `य` and `य़`)
    * Such forms should _not_ be in the layout. They will apply to all layouts of that language, even custom ones.
    * The popup keys will be added to the "_Language_" popup key group (relevant for setting popup key order).
      * Use `%` to mark all preceding keys as "_Language (important)_" instead. Keys after `%` will still be in the "_Language_" group.
    * The `punctuation` key is typically the period key. `popup_keys` set here override the default.
  * `[labels]` may contain non-default labels for the following keys `symbol`, `alphabet`, `shift_symbol`, `shift_symbol_tablet`, `comma`, `period`, `question`
  * `[number_row]` may contain a custom number row (1-9 and 0 separated by space).
  * `[extra_keys]` are typically keys shown in the default layout of the language. This is currently only used for latin layouts to avoid duplicating layouts for just adding few keys on the right side. The layout name need to end with `+`, but the `+` is removed when looking up the actual layout.
* If you add a new language for which Android does not have a display name, it will be displayed using the language tag
  * To fix this, add the language to [`strings.xml`](/app/src/main/res/values/strings.xml) (common names are `subtype_<langage tag>`), and then in [LocaleUtils.localizedDisplayName](/app/src/main/java/helium314/keyboard/latin/common/LocaleUtils.kt#L179) add an `overrideRedId` for the language tag. For overriding the language name displayed on space bar, you need to add the language tag to `subtype_locale_displayed_in_root_locale`, and the name to `subtype_locale_displayed_in_root_locale_display_names` in [donottranslate.xml](/app/src/main/res/values/donottranslate.xml).
* If a newly added language does not use latin script, please update the default scripts method `Locale.script` in [ScriptUtils](app/src/main/java/helium314/keyboard/latin/utils/ScriptUtils.kt)

## Functional key layouts
Customizing functional keys mostly works like other layouts, with some specific adjustments:
* When using the default functional layout, emoji, language switch and numpad keys are actually always in the layout, but get removed depending on settings and the main layout (alphabet, symbols or more symbols). This removal is disabled when you customize any functional layout, so to not block you from adding e.g. a numpad key in alphabet layout.
* When you use a language that has a ZWNJ key, the key will automatically be added to the right of the (first) space bar in the bottom row
* Adding popups to keys that switch layout does not work properly, as usually the layout is switched as soon as the key gets pressed.
* use keys with `"type": "placeholder"` for
  * separating left and right functional keys (e.g. shift and delete in default layout)
  * separating top and bottom rows in case you want to have functional key rows aligned to the top of the keyboard (add a row with the placeholder as the only key)
* if the last row in functional keys does not contain a placeholder, it is used as bottom row (like in the default functional layout)
* When you functional keys only for some of alphabet, symbols and more symbols, behavior is as follows
  * more symbols will fall back to symbols, then normal
  * symbols will fall back to normal, then default (if you only customized more symbols functional layout)
  * normal will fall back to default (if you only customized symbols and/or more symbols functional layout)
