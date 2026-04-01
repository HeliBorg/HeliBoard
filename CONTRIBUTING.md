# Getting Started

HeliBoard project is based on Gradle and Android Gradle Plugin. To get started, you can install [Android Studio](https://developer.android.com/studio), and import project 'from Version Control / Git / Github' by providing this git repository [URL](https://github.com/HeliBorg/HeliBoard) (or git SSH [URL](git@github.com:Helium314/heliboard.git)).
Of course you can also use any other compatible IDE, or work with text editor and command line.
Once everything is up correctly, you're ready to go!

If you have difficulties implementing some functionality, you're welcome to ask for help. No one will write the code for you, but often other contributors can give you very useful hints.

# About the Code

HeliBoard is based on AOSP keyboard, and in many places still contains mostly the original code. There are some extensions, and some parts have been replaced completely.
When working on this app, you will likely notice its rather large size, and quite different code styles and often ancient comments and _TODO_s, where the latter are typically untouched since AOSP times.
Unfortunately a lot of the old code is hard to read or to fully understand with all of its intended (and unintended) consequences.

Some hints for finding what you're looking for:
* Layouts: stored in `layouts` folder in assets, interpreted by `KeyboardParser` and `TextKeyData`
  * Popups: either on layouts, or in `locale_key_texts` (mostly letter variations for specific languages that are not dependent on layout)
* Touch and swipe input handling: `PointerTracker`
* Handling of key inputs: `InputLogic`
* Suggestions: `DictionaryFacilitatorImpl`, `Suggest`, `InputLogic`, and `SuggestionStripView` (in order from creation to display)
* Forwarding entered text / keys to the app / text field: `RichInputConnection`
* Receiving events and information from the app / text field: `LatinIME`
* Settings are in `SettingsValues`, with some functionality in `Settings` and the default values in `Default`

# Guidelines

## Recommended

If you want to contribute, it's a good idea to make sure your idea is actually wanted in HeliBoard.
Best check related issues before you start working on a PR. If the issue has the [labels](https://github.com/HeliBorg/HeliBoard/labels) [_PR_](https://github.com/HeliBorg/HeliBoard/labels/PR) or [_contributor needed_](https://github.com/HeliBorg/HeliBoard/issues?q=label%3A%22contributor%20needed%22) (even closed ones), contributions are wanted. If you don't find a related issue, it's recommended to open one, but ultimately it's your choice.
Asking before starting a PR may help you for getting pointers to potentially relevant code, and deciding how to implement your desired changes.

HeliBoard is a complex application and used by users with a large variety of opinions on how things should be.
When contributing to the app, please:
* Be careful when modifying core components, as it's easy to trigger unintended consequences
* When introducing a feature or change that might not be wanted by everyone, make it optional
* Keep code simple where possible. Complex code is harder to review and to maintain, so the complexity should also add a clear benefit
* Avoid noticeable performance impact. Some parts of the code are executed very frequently, and the keyboard should stay responsive even on older devices.
* Try making use of in-place mechanisms instead of re-inventing the wheel. Your contribution should only add as much complexity as necessary, the code is overly complicated already 😶.
* Keep your changes to few places, as opposed to sprinkling them over many parts of the code. This helps with keeping down complexity during review, and with maintainability of the app.
* Make a draft PR when you intend to still work on it. Submitting an unfinished PR can be a good idea when you're not sure how to best continue and would like some comments.

Further things to consider (though irrelevant for most PRs):
* APK size:
  * Large increases should be discussed first, and will only be added when it's considered worth the increase for a majority of users. It might be possible to avoid size increase by importing optional parts, like it's done for dictionaries.
  * Small increases like when adding code or layouts are never an issue
* Do not add proprietary code or binary blobs. If it turns out to be necessary for a feature you want to add, it might be acceptable when the user opts in and imports those parts, like it's done for glide typing.
* Privacy: Only relevant when adding some form of communication with other apps. Internet permission will not be added.
* If your contribution contains code that is not your own, provide a link to the source
  * This is especially relevant to be sure the code's license is compatible to HeliBoard's GPL3

## Necessary

Some parts of the guidelines are necessary to fulfill for facilitating code review. It doesn't need to be perfect from the start, but consider it for your future PRs when you're reminded of these guidelines. Note that the larger / more complex your PR is, the more relevant these guidelines are.
Your PR should:
- **Be only about a single thing**. Mixing unrelated or semi-related contributions into a single PR is hard to review and can get messy. As a general rule: if one part doesn't need the other one(s), it should be separate PRs. If one feature builds on top of another one, but the base is usable on its own, do a PR for the base and then a follow-up once it's merged.
- **Have a proper description**. A good description helps _a lot_ for understanding what you intend to achieve with the changes, and for understanding the code. This is relevant for separating wanted from unintended changes in behavior during review.
- **No translations**. Translations should be done using [Weblate](https://translate.codeberg.org/projects/heliboard/). Exception is when you add new resource strings, those can be added right away.

Please leave dependency upgrades to the maintainers, unless you state a good reason why they should be done now.

# Adding / Adjusting Layouts

See [layouts.md](layouts.md#adding-new-layouts--languages) for how to add new layouts to the app. Please stay in line with other layouts regarding the popup keys.

When editing existing layouts, please consider that people should should still get what they're used to. In case of doubt it might be better to add a new layout instead of overhauling existing layouts.
`locale_key_texts` files should only contain letters that are actually part of the language, with exception of the optional `more_popups_<...>.txt` files.

# Update Emojis

See make-emoji-keys tool [README](tools/make-emoji-keys/README.md).

# Translations
Translations can be added using [Weblate](https://translate.codeberg.org/projects/heliboard/). You will need an account to update translations and add languages. Add the language you want to translate to in Languages -> Manage translated languages in the top menu bar.
Updating translations in a PR will not be accepted, as it may cause conflicts with Weblate translations.

# Dictionaries
No new dictionaries will be added to this app. Please submit dictionaries and the wordlist to the [dictionaries repository](https://codeberg.org/Helium314/aosp-dictionaries)
