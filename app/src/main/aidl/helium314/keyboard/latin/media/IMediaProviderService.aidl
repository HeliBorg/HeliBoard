package helium314.keyboard.latin.media;

interface IMediaProviderService {
    Bundle discoverCapabilities();
    Bundle search(String query, in Bundle options);
    Bundle browse(String parentId, in Bundle options);
    Bundle getContent(String itemId, in Bundle options);
}
